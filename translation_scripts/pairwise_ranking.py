import pandas as pd
import sys
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the notebooks directory to the path to import common.py
notebooks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
sys.path.append(notebooks_path)
from common import run_rating, create_ranking_prompt, create_prompt_rating, create_edit_prompt, compare_sentences_by_quality, get_response

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform pairwise ranking of translations.')
    
    parser.add_argument('--model', type=str, 
                        default="meta-llama/Llama-3.2-3B-Instruct",
                        help='Model to use for ratings (default: meta-llama/Llama-3.2-3B-Instruct)')
    
    parser.add_argument('--cot', type=lambda x: x.lower() in ['true', '1', 'yes', 'y'], 
                        default=False,
                        help='Use chain-of-thought reasoning (default: False). Set to True/False.')
    
    parser.add_argument('--input_file', type=str,
                        default="/scratch/project_462000353/maribarr/translation_quality/data/Flores200_dev.csv",
                        help='Path to input CSV file')
    
    parser.add_argument('--output_file', type=str,
                        help='Path to output CSV file (default: [input_filename]_rated.csv)')
    
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum completion tokens (default: 1000)')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of comparison repeats (default: 1)')
    
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of examples to process before saving results (default: 10)')
    
    parser.add_argument('--task', type=str, default='rank_vs_edited', 
                        help='score, edit, internal_ranking or rank_vs_edited')
    args = parser.parse_args()
    return args


def process_batch(batch_df, model, tokenizer, args, result_column):
    """Process a batch of examples"""
    # Process each row in the batch
    results = []
    for _, row in batch_df.iterrows():
        if args.task == 'rank_vs_edited':
            result = run_rating(
                translation=row['translations'], 
                correction=row['Correction'], 
                model=model,  # Pass the model object directly
                cot=args.cot,
                rating_prompt_func=create_ranking_prompt, 
                max_completion_tokens=args.max_tokens,
                pipe=None  # No need for a custom pipe
            )
        elif args.task == 'score':
            prompt = create_prompt_rating(row['translations'], cot=args.cot)
            result = get_response(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_completion_tokens=args.max_tokens
            )
        elif args.task == 'edit':
            prompt = create_edit_prompt(row['translations'], cot=args.cot)
            result = get_response(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_completion_tokens=args.max_tokens
            )
        else:
            print(f"Error: Invalid task '{args.task}'. Choose either: score, edit, rank_vs_edited, or internal_ranking")
            sys.exit(1)  # Exit with error code 1 to indicate failure

        results.append(result)
    
    # Add results to the batch dataframe
    batch_df[result_column] = results
    return batch_df

    # Add results to the batch dataframe
    batch_df[result_column] = results
    return batch_df

def main(args=None, pre_loaded_model=None, pre_loaded_tokenizer=None):
    # Parse command-line arguments
    args = args or parse_arguments()
    
    # Load the dataset
    print(f"Loading dataset from {args.input_file}...")
    df = pd.read_csv(args.input_file)
        
    # Display configuration
    print(f"Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Chain of thought: {'Enabled' if args.cot else 'Disabled'}")
    print(f"  - Max tokens: {args.max_tokens}")
    print(f"  - Repeat: {args.repeat}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Output file: {args.output_file}")
    print(f"  - Task: {args.task}")
        
    # Load model and tokenizer using AutoModel - only once (or use pre-loaded ones)
    if pre_loaded_model is not None and pre_loaded_tokenizer is not None:
        print("Using pre-loaded model and tokenizer")
        model = pre_loaded_model
        tokenizer = pre_loaded_tokenizer
    else:
        print(f"Loading model {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    # Initialize result column
    if args.task == 'internal_ranking':
        compare_sentences_by_quality(
            df, 
            id_column='source_sentence', 
            sentence_column='translations', 
            score_column='Fluency_score', 
            model=model,
            tokenizer=tokenizer,  # This is already being passed correctly
            cot=args.cot,
            save_path=args.output_file
        )
        print('Done ranking sentences among themselves')

    elif args.task != 'internal_ranking':
        result_column = "open_model_task_col"
        df[result_column] = None

    # If a partial result file exists, load it
    if os.path.exists(args.output_file):
        print(f"Loading existing results from {args.output_file}...")
        # Check file extension to determine format for reading
        if args.output_file.endswith('.json') or args.output_file.endswith('.jsonl'):
            existing_df = pd.read_json(args.output_file, lines=True, orient='records')
        else:
            # Default to CSV
            existing_df = pd.read_csv(args.output_file)
        
        # Find which rows have already been processed
        processed_mask = ~existing_df["open_model_task_col"].isna()
        
        # Make sure indices match before assignment
        if len(df) == len(existing_df):
            df.loc[processed_mask.index, "open_model_task_col"] = existing_df.loc[processed_mask, "open_model_task_col"]
            print(f"Loaded {processed_mask.sum()} already processed examples")
        else:
            print(f"Warning: Existing file has {len(existing_df)} rows but input file has {len(df)} rows. Attempting to merge...")
            # Use a more robust merge based on a unique identifier if possible
            # For now, just copying over the processed results that exist
            df["open_model_task_col"] = existing_df["open_model_task_col"].values[:len(df)] if len(existing_df) >= len(df) else None
        
        # Define unprocessed indices based on which rows have null values in the result column
        unprocessed_indices = df[df["open_model_task_col"].isna()].index.tolist()
        print(f"Found {len(unprocessed_indices)} unprocessed examples to complete")
    else:
        # If no existing file, process all rows
        unprocessed_indices = df.index.tolist()
        print(f"Processing all {len(unprocessed_indices)} examples")

    # Process in batches with a progress bar
    with tqdm(total=len(unprocessed_indices), desc="Processing examples") as pbar:
        for i in range(0, len(unprocessed_indices), args.batch_size):
            batch_indices = unprocessed_indices[i:i+args.batch_size]
            batch_df = df.loc[batch_indices].copy()
            # Process the batch
            processed_batch = process_batch(batch_df, model, tokenizer, args, result_column)
            
            # Update the main dataframe with the results
            df.loc[batch_indices, result_column] = processed_batch[result_column]
            
            # Save intermediate results
            df.to_json(args.output_file, lines=True, orient='records')
            
            # Update progress bar
            pbar.update(len(batch_indices))
    
    # Save final results
    print(f"Saving final results to {args.output_file}...")
        
    # Check file extension to determine format
    if args.output_file.endswith('.json') or args.output_file.endswith('.jsonl'):
        df.to_json(args.output_file, orient="records", lines=True)
    else:
        # Default to CSV
        df.to_csv(args.output_file, index=False)
        
    print(f"Results saved to {args.output_file}")
    print("Done!")
    del model


    
if __name__ == "__main__":
    main()

    torch.cuda.empty_cache() 