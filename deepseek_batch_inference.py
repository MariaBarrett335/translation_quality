#!/usr/bin/env python3
"""
# DeepSeek Batch Inference Script

A Python script for running parallel batch inference on JSONL datasets using VLLM and DeepSeek models, with automatic resumption, intelligent batching, and robust output parsing.

## Overview

This script processes JSONL input files through a large language model in parallel batches, designed specifically for tasks like Danish fluency rating but adaptable to any structured evaluation task. It includes memory management, automatic batch splitting, and dual parsing strategies for maximum reliability.

## Prerequisites

### Environment Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Ray cluster**: Manually start a Ray cluster with 4 nodes following [Rahul's setup guide](https://docs.google.com/document/d/1EKJmGFTMBzW0RKmvTsueR7VIz9vU1dp3fPepHPL5D7A/edit?tab=t.0#heading=h.yg56rsdbfzwa)
2a. I run the nodes in four tmux sessions
2b. Rahul's run_cluster.sh is already in the repo with the addition that it mounts the current directory to the docker container
3. **Docker container**: Run the script inside the configured Docker environment on the head node:

### Required Files
- **Input file**: JSONL format with configurable column names (default: `prompt` and `id`)
- **Prompt file**: Markdown file containing task instructions and formatting guidelines
- **Output file**: JSONL file for results (supports automatic resumption)

## Quick Start

### Basic Usage
```bash
python deepseek_batch_inference.py \
   data/input_test.jsonl \
   prompts/danish_fluency_prompt.md \
   output_test.jsonl \
   --max-concurrent 20 \
   --items-per-batch 5 \
   --prompt_col sentence


# Notes
- Generation throughput drops rapidly from 100 tokens/s to > 10 tokens/s after an hour or so with the settings above. Maybe due so some memory leakage. I have not solved this. I restart periodically (after 1-2 hours)
- the VLLM KV cache builds up during use  - it goes faster if the output tokens are too long. With < 1000 it can run for a long time
- With too big batches and long generation output, we run into OOM erros
- Some prompts don't follow the expected format and the output is null. I use regex to parse the output. Todo: Improve output parsing robustness. 
- The output parsing assumes the following format with a main output and a reason:
    <id1><o>[main_output]</o><reason>[rationale]</reason></id1>
- The script uses BeautifulSoup to parse the output with fallback to regex but will return None in case the output format is not respected
- If batches are too long, the GPU KV cache builds up. There are two ways to ensure that this does not happen:
    -- items-per-batch max items per batch. If it is too big, the token generation degrades faster
    -- max-tokens This parameter serves as a fallback to items-per-batch and estimates token count from the batch prompt + items-per-batch and splits up the batch if it is too long. Max-tokens above 1500 seems to build up the GPU KV cache
- This script is coded in collaboration with Claude 4
"""
#!/usr/bin/env python3

import json
import argparse
import sys
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid
from bs4 import BeautifulSoup
import time

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
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

def load_prompt_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_items(data: List[Dict[str, Any]], prompt_col: str, id_col: str) -> List[Tuple[str, str]]:
    items = []
    for item in data:
        if prompt_col in item and id_col in item:
            item_id = str(item[id_col])
            prompt_text = str(item[prompt_col]).strip()
            if prompt_text:
                items.append((item_id, prompt_text))
    
    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    for item_id, prompt_text in items:
        key = (item_id, prompt_text)
        if key not in seen:
            seen.add(key)
            deduplicated.append((item_id, prompt_text))
    
    if len(deduplicated) < len(items):
        removed = len(items) - len(deduplicated)
        print(f"Removed {removed} duplicate items ({len(deduplicated)} unique remaining)")
    
    return deduplicated

def load_existing_results(output_file: str) -> set:
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
                        existing_ids.add(str(data['id']))
                except json.JSONDecodeError:
                    continue
    
    if existing_ids:
        print(f"Found {len(existing_ids)} existing results - will resume")
    return existing_ids

def create_task_batches(items: List[Tuple[str, str]], general_prompt: str, 
                       items_per_batch: int = 10, max_tokens: int = 2000, 
                       model_name: str = "unknown") -> List[Dict[str, Any]]:
    batches = []
    instruction_tokens = len(general_prompt) // 4
    
    print(f"Creating batches from {len(items)} items")
    global_batch_counter = 1
    
    for batch_start in range(0, len(items), items_per_batch):
        batch_end = min(batch_start + items_per_batch, len(items))
        batch_items = items[batch_start:batch_end]
        
        # Check if batch needs splitting
        estimated_tokens = instruction_tokens
        for item_id, prompt_text in batch_items:
            estimated_tokens += len(f"<id1-1>{prompt_text}</id1-1>\n") // 4 + 50
        
        debug_print(f"Batch {global_batch_counter}: {len(batch_items)} items, ~{estimated_tokens} tokens")
        
        # Split if too long
        if estimated_tokens > max_tokens:
            print(f"Batch {global_batch_counter} too long ({estimated_tokens} tokens), splitting...")
            available_tokens = max_tokens - instruction_tokens - 300
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for item_id, prompt_text in batch_items:
                item_tokens = len(f"<id1-1>{prompt_text}</id1-1>\n") // 4 + 50
                
                if current_tokens + item_tokens > available_tokens and current_chunk:
                    chunks.append(current_chunk.copy())
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append((item_id, prompt_text))
                current_tokens += item_tokens
            
            if current_chunk:
                chunks.append(current_chunk)
            
            print(f"Split into {len(chunks)} sub-batches")
        else:
            chunks = [batch_items]
        
        # Create batches for each chunk
        for chunk_idx, chunk_items in enumerate(chunks):
            batch_id = f"{global_batch_counter}" if len(chunks) == 1 else f"{global_batch_counter}-{chunk_idx + 1}"
            
            # Create items section with IDs
            items_section = ""
            item_ids = []
            item_texts = []
            
            for i, (item_id, prompt_text) in enumerate(chunk_items):
                simple_id = str(int(item_id))
                item_ids.append(simple_id)
                item_texts.append(prompt_text)
                items_section += f"<id{simple_id}>{prompt_text}</id{simple_id}>\n"
            
            # Create complete prompt with explicit format instructions
            format_instructions = "\nProvide exactly one score and brief rationale for each item:\n"
            for simple_id in item_ids:
                format_instructions += f"<id{simple_id}><o>[score]</o><reason>[rationale]</reason></id{simple_id}>\n"
            
            complete_prompt = general_prompt + "\n\n" + items_section + format_instructions
            final_tokens = len(complete_prompt) // 4
            
            batches.append({
                "batch_id": batch_id,
                "prompt": complete_prompt,
                "item_ids": item_ids,
                "item_texts": item_texts,
                "estimated_tokens": final_tokens,
                "was_split": len(chunks) > 1,
                "model_name": model_name
            })
            
            split_info = " (split)" if len(chunks) > 1 else ""
            print(f"Batch {batch_id}: {len(item_ids)} items, ~{final_tokens} tokens{split_info}")
        
        global_batch_counter += 1
    
    return batches

async def initialize_model(model_path: str, tensor_parallel_size: int = 8, pipeline_parallel_size: int = 4):
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

def parse_task_response_regex(response: str, expected_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    outputs = {}
    reasons = {}
    
    debug_print(f"Using regex parsing for IDs: {expected_ids}")
    
    for expected_id in expected_ids:
        # Primary pattern: <id91><o>2</o><reason>text</reason></id91>
        pattern = rf'<id{re.escape(expected_id)}><o>([^<]+)</o><reason>(.*?)</reason></id{re.escape(expected_id)}>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            outputs[expected_id] = match.group(1).strip()
            reason_text = match.group(2).strip()
            if len(reason_text) > 500:
                reason_text = reason_text[:500] + "..."
            reasons[expected_id] = reason_text
            debug_print(f"Regex matched {expected_id}: '{match.group(1)}'")
        else:
            debug_print(f"No regex match for {expected_id}")
    
    return outputs, reasons

def parse_task_response(response: str, expected_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    outputs = {}
    reasons = {}
    
    debug_print(f"Parsing response for IDs: {expected_ids}")
    
    # Try BeautifulSoup first
    try:
        xml_content = f"<root>{response}</root>"
        soup = BeautifulSoup(xml_content, 'xml')
        
        for expected_id in expected_ids:
            id_tag = soup.find(f"id{expected_id}")
            if id_tag:
                output_tag = id_tag.find('o')
                if output_tag:
                    outputs[expected_id] = output_tag.get_text().strip()
                
                reason_tag = id_tag.find('reason')
                if reason_tag:
                    reason_text = reason_tag.get_text().strip()
                    if len(reason_text) > 500:
                        reason_text = reason_text[:500] + "..."
                    reasons[expected_id] = reason_text
                elif expected_id in outputs:
                    reasons[expected_id] = "No reason provided"
        
        # If BeautifulSoup got all expected IDs, return
        if len(outputs) == len(expected_ids):
            debug_print(f"BeautifulSoup parsed all {len(expected_ids)} IDs")
            return outputs, reasons
            
    except Exception as e:
        debug_print(f"BeautifulSoup failed: {e}")
    
    # Fallback to regex
    debug_print("Using regex fallback")
    return parse_task_response_regex(response, expected_ids)

async def process_batch(engine, batch_data: Dict[str, Any], sampling_params, output_file: str, existing_ids: set) -> Dict[str, Any]:
    batch_id = batch_data['batch_id']
    batch_start_time = time.time()
    
    try:
        print(f"Starting batch {batch_id}")
        
        debug_print(f"Prompt for batch {batch_id}:")
        debug_print(batch_data['prompt'])
        
        # Generate response with timeout
        request_id = random_uuid()
        response = ""
        
        try:
            if hasattr(asyncio, 'timeout'):
                async with asyncio.timeout(1800):
                    async for request_output in engine.generate(batch_data['prompt'], sampling_params, request_id):
                        if request_output.finished and request_output.outputs:
                            response = request_output.outputs[0].text
                            break
            else:
                async def get_response():
                    async for request_output in engine.generate(batch_data['prompt'], sampling_params, request_id):
                        if request_output.finished and request_output.outputs:
                            return request_output.outputs[0].text
                    return ""
                response = await asyncio.wait_for(get_response(), timeout=1800)
        except (asyncio.TimeoutError, asyncio.exceptions.TimeoutError):
            raise Exception(f"Batch timeout after 1800s")
        
        # Parse outputs and reasons
        expected_ids = batch_data['item_ids']
        outputs, reasons = parse_task_response(response, expected_ids)
        
        debug_print(f"Parsed {len(outputs)}/{len(expected_ids)} outputs")

        # Save results
        batch_success_count = 0
        for i, item_id in enumerate(expected_ids):
            if item_id in existing_ids:
                continue
                
            result = {
                'id': item_id,
                'input': batch_data['item_texts'][i],
                'output': outputs.get(item_id, None),
                'reason': reasons.get(item_id, ""),
                'model_name': batch_data['model_name']
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
            
            if result['output'] is not None:
                batch_success_count += 1

        batch_time = time.time() - batch_start_time
        print(f"Completed batch {batch_id} in {batch_time:.1f}s: {len(outputs)}/{len(expected_ids)} outputs")

        # Memory cleanup
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'hip') and torch.hip.is_available():
                torch.hip.empty_cache()
        except:
            pass
        
        return {
            'batch_id': batch_id,
            'outputs_extracted': len(outputs),
            'success': True
        }
        
    except Exception as e:
        batch_time = time.time() - batch_start_time
        print(f"ERROR: Batch {batch_id} failed after {batch_time:.1f}s: {e}")
        
        # Save error entries
        for i, item_id in enumerate(batch_data['item_ids']):
            if item_id in existing_ids:
                continue
                
            error_result = {
                'id': item_id,
                'input': batch_data['item_texts'][i],
                'output': None,
                'reason': "Error during processing",
                'model_name': batch_data['model_name']
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()
        
        gc.collect()
        
        return {
            'batch_id': batch_id,
            'outputs_extracted': 0,
            'success': False
        }

async def run_inference(task_batches: List[Dict[str, Any]], model_path: str, max_concurrent: int, 
                       output_file: str, force_fresh: bool, sampling_params) -> List[Dict[str, Any]]:
    print(f"Starting inference on {len(task_batches)} batches...")
    start_time = time.time()
    
    # Load existing results
    existing_ids = set() if force_fresh else load_existing_results(output_file)
    if force_fresh:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    # Filter batches to process
    def should_process_batch(batch_data):
        return not existing_ids or any(rid not in existing_ids for rid in batch_data['item_ids'])
    
    batches_to_process = [batch for batch in task_batches if should_process_batch(batch)]
    
    if len(batches_to_process) < len(task_batches):
        skipped = len(task_batches) - len(batches_to_process)
        print(f"Skipping {skipped} completed batches, processing {len(batches_to_process)} remaining")
    
    if not batches_to_process:
        print("All batches already completed!")
        return []
    
    # Initialize model and run
    engine = await initialize_model(model_path)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(batch_data):
        async with semaphore:
            return await process_batch(engine, batch_data, sampling_params, output_file, existing_ids)
    
    tasks = [limited_task(batch_data) for batch_data in batches_to_process]
    results = []
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        successful_outputs = sum(r['outputs_extracted'] for r in results if r['success'])
        total_items = sum(len(batch['item_ids']) for batch in batches_to_process)
        progress = successful_outputs / total_items * 100 if total_items > 0 else 100
        print(f"Progress: {len(results)}/{len(tasks)} batches, {successful_outputs}/{total_items} items ({progress:.1f}%)")
    
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    return results

def test_parsing():
    sample_response = """<id91><o>2</o><reason>Mixed Danish/English ("develop"). Grammar error.</reason></id91>
<id92><o>2</o><reason>English words inserted. Awkward structure.</reason></id92>"""
    
    expected_ids = ['91', '92']
    outputs, reasons = parse_task_response(sample_response, expected_ids)
    
    print("=== PARSING TEST ===")
    for id in expected_ids:
        output = outputs.get(id, 'NOT_FOUND')
        reason = reasons.get(id, 'NOT_FOUND')
        print(f"ID {id}: output='{output}', reason='{reason[:50]}...'")
    print("===================")
    
    return len(outputs) == len(expected_ids)

def main():
    global DEBUG
    
    parser = argparse.ArgumentParser(description="DeepSeek batch inference with VLLM")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("prompt_file", help="Markdown file containing the task prompt")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument("--prompt_col", default="prompt", help="Column name for input prompts")
    parser.add_argument("--id_col", default="id", help="Column name for IDs")
    parser.add_argument("--items-per-batch", type=int, default=5, help="Target items per batch")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Max input tokens per batch including the batch prompt. It will clutter the GPU KV cache if too long")
    parser.add_argument("--top-n", type=int, help="Select only top N items")
    parser.add_argument("--model-path", default="/app/model", help="Path to model")
    parser.add_argument("--model-name", default="DeepSeek-R1", help="Model name for output. Only a string. No functionality")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent batches")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--max-generation-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--preview-only", action="store_true", help="Preview without running inference")
    parser.add_argument("--create-batch-file", action="store_true", help="Create batch file for inspection")
    parser.add_argument("--force-fresh", action="store_true", help="Force fresh start")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--test-parsing", action="store_true", help="Test parsing function and exit")
    
    args = parser.parse_args()
    
    DEBUG = args.debug
    
    if args.test_parsing:
        DEBUG = True
        success = test_parsing()
        print(f"Parsing test {'PASSED' if success else 'FAILED'}")
        return
    
    # Load data and prompt
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    general_prompt = load_prompt_file(args.prompt_file)
    items = extract_items(data, args.prompt_col, args.id_col)
    print(f"Found {len(items)} unique items")
    
    if args.top_n:
        items = items[:args.top_n]
        print(f"Selected top {len(items)} items")
    
    # Create batches
    task_batches = create_task_batches(items, general_prompt, args.items_per_batch, 
                                      args.max_tokens, args.model_name)
    total_items = sum(len(batch['item_ids']) for batch in task_batches)
    auto_split_count = sum(1 for batch in task_batches if batch.get('was_split', False))
    
    print(f"\nCreated {len(task_batches)} batches for {total_items} items")
    if auto_split_count > 0:
        print(f"Auto-split: {auto_split_count} batches")
    
    if args.create_batch_file:
        batch_file = args.output_file.replace('.jsonl', '_batches.jsonl')
        with open(batch_file, 'w', encoding='utf-8') as f:
            for batch in task_batches:
                f.write(json.dumps(batch, ensure_ascii=False) + '\n')
        print(f"Batch file saved: {batch_file}")
    
    if args.preview_only:
        print("Preview complete. Remove --preview-only to run inference.")
        return
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        top_k=args.top_k, 
        max_tokens=args.max_generation_tokens
    )
    
    # Run inference
    results = asyncio.run(run_inference(
        task_batches, args.model_path, args.max_concurrent, 
        args.output_file, args.force_fresh, sampling_params
    ))
    
    # Final statistics
    total_outputs = sum(r['outputs_extracted'] for r in results if r['success'])
    total_processed = sum(len(batch['item_ids']) for batch in task_batches)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Successfully processed: {total_outputs}/{total_processed} items ({total_outputs/total_processed*100:.1f}%)")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()