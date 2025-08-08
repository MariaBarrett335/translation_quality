#!/usr/bin/env python3
"""
Process responses from JSONL file and run parallel inference for Danish fluency rating
It is vibe-coded with Claude 4

How to run: 
- Manually start a ray cluster with 4 nodes by following Rahuls instructions: https://docs.google.com/document/d/1EKJmGFTMBzW0RKmvTsueR7VIz9vU1dp3fPepHPL5D7A/edit?tab=t.0#heading=h.yg56rsdbfzwa 
- For each input row, it takes all list items in the responses column and splits the list in batches that fit the specificed size thresholds
- In the docker container, start the script. These settings work well to begin with, then generation token/second slows down:
python batched_script.py data/output_wiki_minp-0-08_topp-1-0_temp-0-1.jsonl output.jsonl --temperature 0.01 --max-concurrent 20 --responses-per-batch 5 --max-tokens 1000
- Generation throughput drops rapidly from 100 tokens/s to > 10 tokens/s after an hour or so. Maybe due so some memory leakage. I have not solved this. I restart periodically (after 1-2 hours)
- the VLLM KV cache builds up during use  - it goes faster if the output tokens are too long. With < 1000 it can run for a long time
- With too big batches and long generation output, we run into OOM erros
- Some prompts don't follow the expected format and the output is null. I use regex to parse the output. Todo: Improve output parsing robustness. 
- If batches are too long, the GPU KV cache builds up. There are two ways to ensure that this does not happen:
    -- items-per-batch max items per batch
    -- max-tokens This parameter serves as a fallback to items-per-batch and estimates token count from the batch prompt + items-per-batch and splits up the batch if it is too long. Max-tokens above 1500 seems to build up the GPU KV cache

Note
- Generation throughput drops rapidly from 100 tokens/s (w 20 concurrent processes) to > 10 tokens/s after an hour or so. Maybe due so some memory leakage. I have not solved this. I restart periodically (after 1-2 hours)
- the VLLM KV cache builds up during use  - it goes faster if the output tokens are too long. With < 1000 it can run for a long time
- With too big batches and long generation output, we run into OOM erros
- Some prompts don't follow the expected format and the output is null. I use regex to parse the output. Todo: Improve output parsing robustness. 
- This script is coded in collaboration with Claude 4

"""

import json
import argparse
import sys
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import gc
import torch
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    print("VLLM not available - will only create batch files")
    VLLM_AVAILABLE = False

import time

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
    return data

def extract_responses_with_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract non-empty responses from the data with their original metadata"""
    all_responses_with_metadata = []
    
    for item in data:
        if 'responses' in item and isinstance(item['responses'], list):
            # Extract the full original column and the paragraph
            original_column = item.get('original', {})
            original_paragraph = ''
            if isinstance(original_column, dict):
                original_paragraph = original_column.get('paragraph', '')
            
            for response in item['responses']:
                if response and response.strip():
                    clean_response = response.strip()
                    all_responses_with_metadata.append({
                        'response': clean_response,
                        'original_paragraph': original_paragraph,
                        'original': original_column
                    })
    
    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    for item in all_responses_with_metadata:
        response = item['response']
        if response not in seen:
            seen.add(response)
            deduplicated.append(item)
    
    if len(deduplicated) < len(all_responses_with_metadata):
        removed = len(all_responses_with_metadata) - len(deduplicated)
        print(f"Removed {removed} duplicate responses ({len(deduplicated)} unique remaining)")
    
    return deduplicated

def load_existing_results(output_file: str) -> set:
    """Load existing IDs from output file to resume processing"""
    existing_ids = set()
    if not Path(output_file).exists():
        return existing_ids
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue
    
    if existing_ids:
        print(f"Found {len(existing_ids)} existing results - will resume")
    return existing_ids

def get_rating_instruction() -> str:
    """Get the rating instruction text"""
    return """Rate Danish language fluency using this detailed rubric:

**1 Point: Minimal Fluency**
- Grammar: Pervasive errors in gender, number, tense, etc.
- Vocabulary: Very basic with repetition and direct translations
- Flow: Choppy, disconnected, extremely unnatural
- Structure: Simple/fragmented sentences, unnatural sequence
- Idioms: No awareness of Danish cultural context

**2 Points: Basic Fluency**  
- Grammar: Frequent errors in articles, verb tenses, meaning understandable
- Vocabulary: Limited with repetition, occasional non-Danish words
- Flow: Distinctly non-Danish cadence, unnatural when read aloud
- Structure: Simple sentences with awkward complexity attempts
- Idioms: Minimal awareness of Danish idioms

**3 Points: Intermediate Fluency**
- Grammar: Some errors with complex structures, meaning clear
- Vocabulary: Adequate for most situations, limited idioms, some repetition
- Flow: Somewhat unnatural rhythm noticeable to natives
- Structure: Mix of simple/complex, reliance on certain patterns
- Idioms: Some awareness but errors like 'jeg bryder problemet ned'

**4 Points: Advanced Fluency**
- Grammar: Very few minor errors, wouldn't distract natives
- Vocabulary: Broad with good idioms, occasional imprecise choices
- Flow: Natural with only occasional awkward phrasing
- Structure: Good variety of complex structures, minor awkwardness
- Idioms: Generally appropriate use, occasional slight misuse

**5 Points: Native-Like Fluency**
- Grammar: Perfect control, no errors in gender/conjugation/word order
- Vocabulary: Rich, precise, idiomatic with Danish expressions/colloquialisms
- Flow: Natural rhythm, completely authentic when read aloud
- Structure: Varied complex structures used appropriately and effortlessly
- Idioms: Appropriate Danish idioms, verbal phrases, Danish-specific expressions

"""

def create_rating_batches(responses_with_metadata: List[Dict[str, Any]], responses_per_batch: int = 10, max_tokens: int = 2000) -> List[Dict[str, Any]]:
    """Create rating batches with automatic splitting when too long"""
    batches = []
    rating_instruction = get_rating_instruction()
    instruction_tokens = len(rating_instruction) // 4
    
    print(f"Creating batches from {len(responses_with_metadata)} responses")
    global_batch_counter = 1
    
    for batch_start in range(0, len(responses_with_metadata), responses_per_batch):
        batch_end = min(batch_start + responses_per_batch, len(responses_with_metadata))
        batch_items = responses_with_metadata[batch_start:batch_end]
        
        # Check if batch needs splitting
        estimated_tokens = instruction_tokens
        for item in batch_items:
            response = item['response']
            estimated_tokens += len(f"<id1-1>{response}</id1-1>\n") // 4 + 50
        
        # Split if too long
        if estimated_tokens > max_tokens:
            print(f"Batch {global_batch_counter} too long ({estimated_tokens} tokens), splitting...")
            available_tokens = max_tokens - instruction_tokens - 300
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for item in batch_items:
                response = item['response']
                response_tokens = len(f"<id1-1>{response}</id1-1>\n") // 4 + 50
                
                if current_tokens + response_tokens > available_tokens and current_chunk:
                    chunks.append(current_chunk.copy())
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(item)
                current_tokens += response_tokens
            
            if current_chunk:
                chunks.append(current_chunk)
            
            print(f"Split into {len(chunks)} sub-batches")
        else:
            chunks = [batch_items]
        
        # Create batches for each chunk
        for chunk_idx, chunk_items in enumerate(chunks):
            batch_id = f"{global_batch_counter}" if len(chunks) == 1 else f"{global_batch_counter}-{chunk_idx + 1}"
            
            # Create response section with IDs
            responses_section = ""
            response_ids = []
            responses_text = []
            responses_metadata = []
            
            for i, item in enumerate(chunk_items):
                response_id = f"{batch_id}-{i + 1}"
                response_ids.append(response_id)
                responses_text.append(item['response'])
                responses_metadata.append({
                    'original_paragraph': item['original_paragraph'],
                    'original': item['original']
                })
                responses_section += f"<id{response_id}>{item['response']}</id{response_id}>\n"
            
            # Create complete prompt
            complete_prompt = rating_instruction + responses_section + "\nProvide exactly one score (1-5) and brief rationale (1-2 sentences) for each response:\n"
            for response_id in response_ids:
                complete_prompt += f"<id{response_id}><score>[1-5]</score></id{response_id}>\n[Brief rationale]\n\n"
            
            final_tokens = len(complete_prompt) // 4
            
            batches.append({
                "batch_id": batch_id,
                "prompt": complete_prompt,
                "response_ids": response_ids,
                "responses_text": responses_text,
                "responses_metadata": responses_metadata,
                "estimated_tokens": final_tokens,
                "was_split": len(chunks) > 1
            })
            
            split_info = " (split)" if len(chunks) > 1 else ""
            print(f"Batch {batch_id}: {len(response_ids)} responses, ~{final_tokens} tokens{split_info}")
        
        global_batch_counter += 1
    
    return batches

async def initialize_model(model_path: str, tensor_parallel_size: int = 8, pipeline_parallel_size: int = 4):
    """Initialize the VLLM model"""
    print("Initializing model...")
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enforce_eager=True,
        max_model_len=4096,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.95
    )
    return AsyncLLMEngine.from_engine_args(engine_args)

def parse_rating_response(response: str, expected_ids: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Parse rating response and extract scores and rationales for each ID"""
    scores = {}
    rationales = {}
    
    for expected_id in expected_ids:
        # Pattern to match <idXXX><score>N</score></idXXX> followed by rationale text
        pattern = rf'<id{re.escape(expected_id)}><score>([1-5])</score></id{re.escape(expected_id)}>\s*\n?(.*?)(?=<id\d+(?:-\d+)*><score>|$)'
        
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    scores[expected_id] = score
                    
                    # Extract and clean rationale (limit to first 200 chars)
                    rationale = match.group(2).strip()
                    rationale = re.sub(r'^\*\*Rationale:\*\*\s*', '', rationale, flags=re.IGNORECASE)
                    rationale = re.sub(r'\*\*(.*?)\*\*', r'\1', rationale)
                    rationale = rationale.strip()
                    
                    if len(rationale) > 200:
                        rationale = rationale[:200] + "..."
                    
                    rationales[expected_id] = rationale
            except ValueError:
                continue
        else:
            # Fallback: try simpler patterns for just the score
            for pattern in [
                rf'<id{re.escape(expected_id)}><score>([1-5])</score>',
                rf'<id{re.escape(expected_id)}>([1-5])</id[^>]*>',
            ]:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for score_match in matches:
                    try:
                        score = int(score_match)
                        if 1 <= score <= 5:
                            scores[expected_id] = score
                            rationales[expected_id] = "No rationale"
                            break
                    except ValueError:
                        continue
                if expected_id in scores:
                    break
    
    return scores, rationales

async def process_batch(engine, batch_data: Dict[str, Any], sampling_params, output_file: str, existing_ids: set) -> Dict[str, Any]:
    """Process a single batch and save results with original data"""
    batch_id = batch_data['batch_id']
    batch_start_time = time.time()
    
    try:
        print(f"Starting batch {batch_id}")
        
        # Generate response with timeout
        request_id = random_uuid()
        response = ""
        
        try:
            if hasattr(asyncio, 'timeout'):
                async with asyncio.timeout(1800):  # 30 min timeout
                    async for request_output in engine.generate(batch_data['prompt'], sampling_params, request_id):
                        if request_output.finished and request_output.outputs:
                            response = request_output.outputs[0].text
                            break
            else:
                # Fallback for older Python versions
                async def get_response():
                    async for request_output in engine.generate(batch_data['prompt'], sampling_params, request_id):
                        if request_output.finished and request_output.outputs:
                            return request_output.outputs[0].text
                    return ""
                response = await asyncio.wait_for(get_response(), timeout=1800)
        except (asyncio.TimeoutError, asyncio.exceptions.TimeoutError):
            raise Exception(f"Batch timeout after 1800s")
        
        # Parse scores and rationales
        expected_ids = batch_data['response_ids']
        scores, rationales = parse_rating_response(response, expected_ids)
        
        # Save results
        batch_success_count = 0
        for i, response_id in enumerate(expected_ids):
            if response_id in existing_ids:
                continue
            
            sentence = batch_data['responses_text'][i]
            metadata = batch_data['responses_metadata'][i]
            
            result = {
                'id': response_id,
                'sentence': sentence,
                'fluency_score': scores.get(response_id, None),
                'reason': rationales.get(response_id, ""),
                'original_paragraph': metadata['original_paragraph'],
                'matched_response': sentence,
                'original': metadata['original']
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
            
            if result['fluency_score'] is not None:
                batch_success_count += 1
        
        batch_time = time.time() - batch_start_time
        print(f"Completed batch {batch_id} in {batch_time:.1f}s: {len(scores)}/{len(expected_ids)} scores")
        # In parse_rating_response() function, add at the top:
        print(f"DEBUG: Raw response for batch {expected_ids}:")
        print(response[:500])  # First 500 chars
        print("="*50)

        gc.collect()  # Python garbage collection
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'hip') and torch.hip.is_available():
                torch.hip.empty_cache()
        except:
            pass
        
        return {
            'batch_id': batch_id,
            'scores_extracted': len(scores),
            'success': True
        }
        
    except Exception as e:
        batch_time = time.time() - batch_start_time
        print(f"ERROR: Batch {batch_id} failed after {batch_time:.1f}s: {e}")
        
        # Save error entries
        for i, response_id in enumerate(batch_data['response_ids']):
            if response_id in existing_ids:
                continue
            
            sentence = batch_data['responses_text'][i]
            metadata = batch_data['responses_metadata'][i]
                
            error_result = {
                'id': response_id,
                'sentence': sentence,
                'fluency_score': None,
                'reason': "Error during processing",
                'original_paragraph': metadata['original_paragraph'],
                'matched_response': sentence,
                'original': metadata['original']
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()
        gc.collect()  # Python garbage collection
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'hip') and torch.hip.is_available():
                torch.hip.empty_cache()
        except:
            pass
        
        return {
            'batch_id': batch_id,
            'scores_extracted': 0,
            'success': False
        }
    

async def run_inference(rating_batches: List[Dict[str, Any]], model_path: str, max_concurrent: int, 
                       output_file: str, force_fresh: bool) -> List[Dict[str, Any]]:
    """Run parallel inference on rating batches"""
    print(f"Starting inference on {len(rating_batches)} batches...")
    start_time = time.time()
    
    # Load existing results
    existing_ids = set() if force_fresh else load_existing_results(output_file)
    if force_fresh:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # Create empty file
    
    # Filter batches to process
    def should_process_batch(batch_data):
        return not existing_ids or any(rid not in existing_ids for rid in batch_data['response_ids'])
    
    batches_to_process = [batch for batch in rating_batches if should_process_batch(batch)]
    
    if len(batches_to_process) < len(rating_batches):
        skipped = len(rating_batches) - len(batches_to_process)
        print(f"Skipping {skipped} completed batches, processing {len(batches_to_process)} remaining")
    
    if not batches_to_process:
        print("All batches already completed!")
        return []
    
    # Initialize model and run
    engine = await initialize_model(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=2048)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(batch_data):
        async with semaphore:
            return await process_batch(engine, batch_data, sampling_params, output_file, existing_ids)
    
    tasks = [limited_task(batch_data) for batch_data in batches_to_process]
    results = []
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        successful_scores = sum(r['scores_extracted'] for r in results if r['success'])
        total_responses = sum(len(batch['response_ids']) for batch in batches_to_process)
        progress = successful_scores / total_responses * 100 if total_responses > 0 else 100
        print(f"Progress: {len(results)}/{len(tasks)} batches, {successful_scores}/{total_responses} responses ({progress:.1f}%)")
    
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract responses and run Danish fluency rating with auto-splitting")
    parser.add_argument("input_file", help="Input JSONL file with responses")
    parser.add_argument("output_file", help="Output JSONL file for scores")
    parser.add_argument("--responses-per-batch", type=int, default=20, help="Target responses per batch (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max tokens per batch (auto-split if exceeded) (default: 3000)")
    parser.add_argument("--top-n", type=int, help="Select only top N responses")
    parser.add_argument("--model-path", default="/app/model", help="Path to model")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent batches (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--max-generation-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--preview-only", action="store_true", help="Preview without running inference")
    parser.add_argument("--create-batch-file", action="store_true", help="Create batch file for inspection")
    parser.add_argument("--force-fresh", action="store_true", help="Force fresh start, ignore existing results")
    
    args = parser.parse_args()
    
    # Load and extract responses
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    responses_with_metadata = extract_responses_with_metadata(data)
    print(f"Found {len(responses_with_metadata)} unique responses")
    
    if args.top_n:
        responses_with_metadata = responses_with_metadata[:args.top_n]
        print(f"Selected top {len(responses_with_metadata)} responses")
    
    # Create batches
    rating_batches = create_rating_batches(responses_with_metadata, args.responses_per_batch, args.max_tokens)
    total_responses = sum(len(batch['response_ids']) for batch in rating_batches)
    auto_split_count = sum(1 for batch in rating_batches if batch.get('was_split', False))
    
    print(f"\nCreated {len(rating_batches)} batches for {total_responses} responses")
    if auto_split_count > 0:
        print(f"Auto-split: {auto_split_count} batches")
    
    if args.create_batch_file:
        batch_file = args.output_file.replace('.jsonl', '_batches.jsonl')
        with open(batch_file, 'w', encoding='utf-8') as f:
            for batch in rating_batches:
                f.write(json.dumps(batch, ensure_ascii=False) + '\n')
        print(f"Batch file saved: {batch_file}")
    
    if args.preview_only:
        print("Preview complete. Remove --preview-only to run inference.")
        return
    
    if not VLLM_AVAILABLE:
        print("VLLM not available for inference")
        return
    
    # Run inference
    results = asyncio.run(run_inference(
        rating_batches, args.model_path, args.max_concurrent, 
        args.output_file, args.force_fresh
    ))
    
    # Final statistics
    total_scores = sum(r['scores_extracted'] for r in results if r['success'])
    total_processed = sum(len(batch['response_ids']) for batch in rating_batches)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Successfully rated: {total_scores}/{total_processed} responses ({total_scores/total_processed*100:.1f}%)")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()