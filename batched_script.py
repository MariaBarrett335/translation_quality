#!/usr/bin/env python3
"""
Process responses from JSONL file and run parallel inference for Danish fluency rating
Creates batches with proper format and runs inference directly
AUTOMATICALLY SPLITS BATCHES WHEN TOO LONG
Disclaimer: It is vibe-coded with Claude Sonnet
"""

import json
import argparse
import sys
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Only import VLLM if actually running inference
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

def extract_responses(data: List[Dict[str, Any]]) -> List[str]:
    """Extract non-empty responses from the data"""
    all_responses = []
    for item in data:
        if 'responses' in item and isinstance(item['responses'], list):
            for response in item['responses']:
                if response and response.strip():
                    all_responses.append(response.strip())
    return all_responses

def deduplicate_responses(responses: List[str]) -> List[str]:
    """Remove duplicate responses while preserving order"""
    seen = set()
    deduplicated = []
    
    for response in responses:
        if response not in seen:
            seen.add(response)
            deduplicated.append(response)
    
    if len(deduplicated) < len(responses):
        removed = len(responses) - len(deduplicated)
        print(f"Removed {removed} duplicate responses ({len(deduplicated)} unique remaining)")
    
    return deduplicated

def load_existing_results(output_file: str) -> set:
    """Load existing IDs from output file to resume processing"""
    existing_ids = set()
    
    if not Path(output_file).exists():
        return existing_ids
    
    try:
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
    except Exception as e:
        print(f"Error reading existing output: {e}")
        return set()
    
    if existing_ids:
        print(f"Found {len(existing_ids)} existing results - will resume")
    
    return existing_ids

def deduplicate_output_file(output_file: str):
    """Remove duplicate entries from output file, keeping the last occurrence"""
    if not Path(output_file).exists():
        return
    
    seen_ids = {}  # id -> line_data mapping
    
    try:
        # Read all lines and keep track of last occurrence of each ID
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            seen_ids[data['id']] = data
                    except json.JSONDecodeError:
                        continue
        
        # Write back deduplicated results
        if seen_ids:
            with open(output_file, 'w', encoding='utf-8') as f:
                for data in seen_ids.values():
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            print(f"Deduplicated output file: {len(seen_ids)} unique results")
    
    except Exception as e:
        print(f"Error during deduplication: {e}")

def should_process_batch(batch_data: Dict[str, Any], existing_ids: set) -> bool:
    """Check if batch has any unprocessed responses"""
    if not existing_ids:
        return True
    
    for response_id in batch_data['response_ids']:
        if response_id not in existing_ids:
            return True
    return False

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (1 token ≈ 4 characters for most languages)"""
    return len(text) // 4

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

def split_responses_by_token_limit(responses: List[str], max_tokens: int = 2000) -> List[List[str]]:
    """Split responses into chunks that fit within token limits"""
    rating_instruction_tokens = estimate_tokens(get_rating_instruction())
    available_tokens = max_tokens - rating_instruction_tokens - 300  # Safety margin
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for response in responses:
        response_tokens = estimate_tokens(f"<id1-1>{response}</id1-1>\n") + 50
        
        if current_tokens + response_tokens > available_tokens and current_chunk:
            chunks.append(current_chunk.copy())
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(response)
        current_tokens += response_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_rating_batches_with_auto_split(responses: List[str], 
                                        responses_per_batch: int = 10,  # Reduced from 20
                                        max_tokens: int = 2000) -> List[Dict[str, Any]]:  # Reduced from 3000
    """Create rating batches with automatic splitting when too long"""
    batches = []
    filtered_responses = [r for r in responses if r and r.strip()]
    
    if not filtered_responses:
        return batches
    
    print(f"Creating batches from {len(filtered_responses)} responses")
    print(f"Target: {responses_per_batch} responses/batch, max {max_tokens} tokens")
    
    rating_instruction = get_rating_instruction()
    global_batch_counter = 1
    
    for batch_start in range(0, len(filtered_responses), responses_per_batch):
        batch_end = min(batch_start + responses_per_batch, len(filtered_responses))
        initial_batch_responses = filtered_responses[batch_start:batch_end]
        
        # Check if batch needs splitting
        estimated_tokens = estimate_tokens(rating_instruction)
        for response in initial_batch_responses:
            estimated_tokens += estimate_tokens(f"<id1-1>{response}</id1-1>\n") + 50
        
        if estimated_tokens <= max_tokens:
            response_chunks = [initial_batch_responses]
        else:
            print(f"Batch {global_batch_counter} too long ({estimated_tokens} tokens), splitting...")
            response_chunks = split_responses_by_token_limit(initial_batch_responses, max_tokens)
            print(f"Split into {len(response_chunks)} sub-batches")
        
        # Create batches for each chunk
        for chunk_idx, chunk_responses in enumerate(response_chunks):
            if not chunk_responses:
                continue
            
            batch_id = f"{global_batch_counter}" if len(response_chunks) == 1 else f"{global_batch_counter}-{chunk_idx + 1}"
            
            # Create response section with IDs
            responses_section = ""
            response_ids = []
            
            for i, response in enumerate(chunk_responses):
                if not response or not response.strip():
                    continue
                response_id = f"{batch_id}-{i + 1}"
                response_ids.append(response_id)
                responses_section += f"<id{response_id}>{response}</id{response_id}>\n"
            
            if not response_ids:
                continue
            
            # Create complete prompt
            complete_prompt = rating_instruction + responses_section + "\nProvide exactly one score (1-5) for each response:\n"
            for response_id in response_ids:
                complete_prompt += f"<id{response_id}><score>[1-5]</score></id{response_id}>\n"
            
            final_tokens = estimate_tokens(complete_prompt)
            
            batches.append({
                "batch_id": batch_id,
                "prompt": complete_prompt,
                "response_ids": response_ids,
                "num_responses_in_batch": len(response_ids),
                "responses_text": chunk_responses[:len(response_ids)],
                "estimated_tokens": final_tokens,
                "was_split": len(response_chunks) > 1
            })
            
            split_info = " (split)" if len(response_chunks) > 1 else ""
            print(f"Batch {batch_id}: {len(response_ids)} responses, ~{final_tokens} tokens{split_info}")
        
        global_batch_counter += 1
    
    return batches

async def initialize_model(model_path: str, 
                          tensor_parallel_size: int = 8,
                          pipeline_parallel_size: int = 4,
                          max_model_len: int = 4096,
                          gpu_memory_utilization: float = 0.95):
    """Initialize the VLLM model"""
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM not available")
    
    print("Initializing model...")
    
    try:
        if pipeline_parallel_size > 1:
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enforce_eager=True,
                max_model_len=max_model_len,
                distributed_executor_backend="ray",
                gpu_memory_utilization=gpu_memory_utilization
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            return engine, True
        else:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=True,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization
            )
            return llm, False
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

def create_sampling_params(temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, max_tokens: int = 2048):
    """Create sampling parameters"""
    if not VLLM_AVAILABLE:
        return None
    return SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)

async def process_single_batch_with_save(engine, batch_data: Dict[str, Any], sampling_params, 
                                      request_id: str, output_file: str, existing_ids: set) -> Dict[str, Any]:
    """Process a single batch and save results immediately"""
    batch_id = batch_data['batch_id']
    batch_start_time = time.time()
    
    try:
        print(f"Starting batch {batch_id} (~{batch_data.get('estimated_tokens', 'unknown')} tokens)")
        
        response = ""
        timeout_seconds = 1800  # 30 minutes timeout per batch
        
        try:
            # Handle timeout for different Python versions
            if hasattr(asyncio, 'timeout'):
                # Python 3.11+
                async with asyncio.timeout(timeout_seconds):
                    async for request_output in engine.generate(batch_data['prompt'], sampling_params, request_id):
                        if request_output.finished:
                            if request_output.outputs:
                                response = request_output.outputs[0].text
                            break
            else:
                # Python 3.10 and earlier
                async with asyncio.wait_for(
                    _get_response_from_engine(engine, batch_data['prompt'], sampling_params, request_id),
                    timeout=timeout_seconds
                ):
                    response = await _get_response_from_engine(engine, batch_data['prompt'], sampling_params, request_id)
        except (asyncio.TimeoutError, asyncio.exceptions.TimeoutError):
            print(f"TIMEOUT: Batch {batch_id} exceeded {timeout_seconds}s, marking as failed")
            raise Exception(f"Batch timeout after {timeout_seconds}s")
        
        batch_time = time.time() - batch_start_time
        print(f"Completed batch {batch_id} in {batch_time:.1f}s")
        
        # Parse scores and save only new results
        expected_ids = batch_data['response_ids']
        scores = parse_rating_response(response, expected_ids)
        
        batch_success_count = 0
        skipped_count = 0
        
        for i, response_id in enumerate(expected_ids):
            # Skip if already processed
            if response_id in existing_ids:
                skipped_count += 1
                continue
                
            score = scores.get(response_id, None)
            original_text = batch_data['responses_text'][i] if i < len(batch_data['responses_text']) else ""
            
            result = {
                'id': response_id,
                'sentence': original_text,
                'fluency_score': score
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
            
            if score is not None:
                batch_success_count += 1
        
        if skipped_count > 0:
            print(f"Batch {batch_id}: {len(scores)}/{len(expected_ids)} scores extracted, {skipped_count} already existed")
        else:
            print(f"Batch {batch_id}: {len(scores)}/{len(expected_ids)} scores extracted")
        
        return {
            'batch_id': batch_id,
            'scores_extracted': len(scores),
            'responses_processed': len(expected_ids),
            'skipped_existing': skipped_count,
            'processing_time': batch_time,
            'success': True
        }
        
    except Exception as e:
        batch_time = time.time() - batch_start_time
        print(f"ERROR: Batch {batch_id} failed after {batch_time:.1f}s: {e}")
        
        # Save error entries for new responses only
        for i, response_id in enumerate(batch_data['response_ids']):
            if response_id in existing_ids:
                continue
                
            original_text = batch_data['responses_text'][i] if i < len(batch_data['responses_text']) else ""
            error_result = {
                'id': response_id,
                'sentence': original_text,
                'fluency_score': None
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()
        
        return {
            'batch_id': batch_id,
            'scores_extracted': 0,
            'responses_processed': len(batch_data['response_ids']),
            'processing_time': batch_time,
            'success': False
        }

async def _get_response_from_engine(engine, prompt, sampling_params, request_id):
    """Helper function for older Python versions without asyncio.timeout"""
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        if request_output.finished:
            if request_output.outputs:
                return request_output.outputs[0].text
            break
    return ""

async def run_parallel_inference(rating_batches: List[Dict[str, Any]], 
                                model_path: str = "/app/model",
                                max_concurrent: int = 3,
                                temperature: float = 0.7,
                                top_p: float = 0.95,
                                top_k: int = 40,
                                max_tokens: int = 2048,
                                output_file: str = "scores.jsonl",
                                force_fresh: bool = False) -> List[Dict[str, Any]]:
    """Run parallel inference on rating batches"""
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM not available for inference")
    
    print(f"Starting inference on {len(rating_batches)} batches...")
    start_time = time.time()
    
    # Load existing results for resume capability (with automatic deduplication)
    existing_ids = set()
    if not force_fresh:
        # Always deduplicate existing file before loading results
        if Path(output_file).exists():
            deduplicate_output_file(output_file)
        existing_ids = load_existing_results(output_file)
    
    # If no existing results or forced fresh start, initialize clean output file
    if not existing_ids or force_fresh:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        if force_fresh:
            print(f"Force fresh start: cleared output file")
        else:
            print(f"Initialized clean output file: {output_file}")
        existing_ids = set()  # Ensure empty set for fresh start
    
    # Filter batches to only process those with unprocessed responses
    batches_to_process = [batch for batch in rating_batches if should_process_batch(batch, existing_ids)]
    
    if len(batches_to_process) < len(rating_batches):
        skipped = len(rating_batches) - len(batches_to_process)
        print(f"Skipping {skipped} fully completed batches, processing {len(batches_to_process)} remaining")
    
    if not batches_to_process:
        print("All batches already completed!")
        return []
    
    # Initialize model
    engine, is_async = await initialize_model(model_path)
    if not is_async:
        print("Error: Pipeline parallelism required")
        sys.exit(1)
    
    sampling_params = create_sampling_params(temperature, top_p, top_k, max_tokens)
    
    # Run with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(batch_data):
        async with semaphore:
            request_id = random_uuid()
            return await process_single_batch_with_save(
                engine, batch_data, sampling_params, request_id, output_file, existing_ids
            )
    
    tasks = [limited_task(batch_data) for batch_data in batches_to_process]
    results = []
    
    completed = 0
    successful_scores = 0
    total_responses = sum(batch['num_responses_in_batch'] for batch in batches_to_process)
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        
        if result['success']:
            successful_scores += result['scores_extracted']
            progress = successful_scores / total_responses * 100 if total_responses > 0 else 100
            print(f"Progress: {completed}/{len(tasks)} batches, {successful_scores}/{total_responses} responses ({progress:.1f}%)")
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    return results

def parse_rating_response(response: str, expected_ids: List[str]) -> Dict[str, int]:
    """Parse rating response and extract scores for each ID"""
    scores = {}
    
    for expected_id in expected_ids:
        patterns = [
            rf'<id{re.escape(expected_id)}><score>([1-5])</score>',
            rf'<id{re.escape(expected_id)}>([1-5])</id[^>]*>',
            rf'id{re.escape(expected_id)}\s*:\s*([1-5])',
            rf'id{re.escape(expected_id)}\s+([1-5])(?:\s|$|[^\d])',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    score = int(match)
                    if 1 <= score <= 5:
                        scores[expected_id] = score
                        break
                except ValueError:
                    continue
            if expected_id in scores:
                break
    
    return scores

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
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Load and extract responses
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    if not data:
        print("Error: No valid data found")
        sys.exit(1)
    
    responses = extract_responses(data)
    print(f"Found {len(responses)} responses")
    
    # Automatically deduplicate responses
    responses = deduplicate_responses(responses)
    
    if args.top_n:
        responses = responses[:args.top_n]
        print(f"Selected top {len(responses)} responses")
    
    # Create batches with auto-splitting
    rating_batches = create_rating_batches_with_auto_split(
        responses, args.responses_per_batch, args.max_tokens
    )
    
    total_responses = sum(batch['num_responses_in_batch'] for batch in rating_batches)
    auto_split_count = sum(1 for batch in rating_batches if batch.get('was_split', False))
    
    print(f"\nCreated {len(rating_batches)} batches for {total_responses} responses")
    if auto_split_count > 0:
        print(f"Auto-split: {auto_split_count} batches")
    
    if args.create_batch_file:
        batch_file = args.output_file.replace('.jsonl', '_batches.jsonl')
        save_jsonl(rating_batches, batch_file)
        print(f"Batch file saved: {batch_file}")
    
    if args.preview_only:
        print("Preview complete. Remove --preview-only to run inference.")
        return
    
    if not VLLM_AVAILABLE:
        print("VLLM not available. Create batch file with --create-batch-file")
        return
    
    # Run inference
    async def run_inference():
        return await run_parallel_inference(
            rating_batches,
            model_path=args.model_path,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_generation_tokens,
            output_file=args.output_file,
            force_fresh=args.force_fresh
        )
    
    results = asyncio.run(run_inference())
    
    # Final statistics
    total_scores = sum(r['scores_extracted'] for r in results if r['success'])
    total_processed = sum(r['responses_processed'] for r in results)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Successfully rated: {total_scores}/{total_processed} responses ({total_scores/total_processed*100:.1f}%)")
    
    if auto_split_count > 0:
        print(f"Auto-split batches: {auto_split_count}")
    
    # Analyze score distribution
    try:
        with open(args.output_file, 'r', encoding='utf-8') as f:
            scores = []
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('fluency_score') is not None:
                        scores.append(data['fluency_score'])
                except:
                    continue
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average score: {avg_score:.2f}")
            print("Distribution:")
            for score in range(1, 6):
                count = scores.count(score)
                pct = count / len(scores) * 100
                print(f"  Score {score}: {count} ({pct:.1f}%)")
    except Exception as e:
        print(f"Could not analyze scores: {e}")
    
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()