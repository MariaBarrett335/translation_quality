# Alternative approach: Collect data first, then create DataFrame at once
import tqdm
import fireducks.pandas as pd
import io
import numpy as np
import glob
from datasets import load_dataset
import argparse
import sys

# samples nsamples from the clean wiki evenly distributed from all files. 
# Each sample consists of three consecutive sentences (unless the sample is the first or last sentence in an article)

def build_fewshot_prompt(fewshot_examples: list, source_label="English", target_label="Finnish") -> str:
    """
    Build a prompt with a header and few-shot examples.
    Each example is formatted as:
    <source_label>: <example source sentence>
    <target_label>: <example target sentence>
    The test sentence is appended with the target cue.
    """
    prompt = f"Translate the following {source_label} sentences to {target_label}.\n\n"
    for ex in fewshot_examples:
        prompt += f"{source_label}: {ex['source']}\n{target_label}: {ex['target']}\n\n"
    return prompt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Wikipedia data for translation tasks')
    
    # Data processing parameters
    parser.add_argument('--nsamples', type=int, default=12000,
                        help='Number of samples to extract (default: 12000)')
    parser.add_argument('--parquet_pattern', type=str, default='data/wiki_batch_*.parquet',
                        help='Glob pattern for parquet files (default: data/wiki_batch_*.parquet)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of parquet files to process (default: all)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file name (default: <count>_clean_wiki.jsonl)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Translation parameters
    parser.add_argument('--source_lang', type=str, default='eng_Latn',
                        help='Source language code (default: eng_Latn)')
    parser.add_argument('--target_lang', type=str, default='dan_Latn',
                        help='Target language code (default: dan_Latn)')
    parser.add_argument('--source_label', type=str, default='English',
                        help='Source language label (default: English)')
    parser.add_argument('--target_label', type=str, default='Danish',
                        help='Target language label (default: Danish)')
    parser.add_argument('--fewshot_num', type=int, default=10,
                        help='Number of few-shot examples (default: 10)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Get parquet files
    parquetfiles = glob.glob(args.parquet_pattern)
    
    # Check if any files were found
    if not parquetfiles:
        print(f"Error: No files found matching pattern '{args.parquet_pattern}'")
        sys.exit(1)
    
    # Apply max_files limit if specified
    if args.max_files:
        parquetfiles = parquetfiles[:args.max_files]
    
    # Calculate samples per file
    n_from_each = int(args.nsamples / len(parquetfiles))
    print(f"Found {len(parquetfiles)} files, targeting {n_from_each} samples per file")

    all_samples = []

    for f in tqdm.tqdm(parquetfiles, desc="Processing files"):
        try:
            # Explicitly set dtype when reading parquet
            df = pd.read_parquet(f)
            
            # Convert sentence column to string explicitly
            if 'sentence' in df.columns:
                df['sentence'] = df['sentence'].astype(str)
                
            grouped = df.groupby('group_id')
            n_groups = len(grouped)
            n_from_each_group = max(1, int(n_from_each / n_groups))
            
            for group_id, group_df in grouped:
                # Skip empty groups
                if len(group_df) == 0:
                    continue
                    
                # Sample rows from this group
                if len(group_df) <= n_from_each_group:
                    # Take all rows if we have fewer than needed
                    sampled_indices = group_df.index.tolist()
                else:
                    # Sample randomly without replacement
                    sampled_indices = np.random.choice(group_df.index, 
                                                    size=n_from_each_group, 
                                                    replace=False).tolist()
                
                # Process each sampled row to include context
                for idx in sampled_indices:
                    try:
                        # Get the current row data as a dictionary
                        row_data = group_df.loc[idx].to_dict()
                        
                        # Find index position within the group
                        group_indices = group_df.index.tolist()
                        pos_in_group = group_indices.index(idx)
                        
                        # Get previous sentence if available within the same group
                        if pos_in_group > 0:
                            prev_idx = group_indices[pos_in_group - 1]
                            row_data['prev_sentence'] = str(group_df.loc[prev_idx, 'sentence'])
                        else:
                            row_data['prev_sentence'] = None
                            
                        # Get next sentence if available within the same group
                        if pos_in_group < len(group_indices) - 1:
                            next_idx = group_indices[pos_in_group + 1]
                            row_data['next_sentence'] = str(group_df.loc[next_idx, 'sentence'])
                        else:
                            row_data['next_sentence'] = None
                        
                        # Add to our samples list
                        all_samples.append(row_data)
                        
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                        continue  # Skip this row and continue with the next
        
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            continue  # Skip this file and continue with the next

    # Create DataFrame from collected samples
    try:
        # Check if we collected any samples
        if not all_samples:
            print("Error: No samples were collected.")
            sys.exit(1)
            
        df = pd.DataFrame(all_samples)
        
        # Shuffle the final sample
        df = df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        
        # Create paragraph column by combining previous, current, and next sentences
        df['paragraph'] = df.apply(
            lambda row: ' '.join(filter(None, [
                row['prev_sentence'] if pd.notna(row['prev_sentence']) else '',
                row['sentence'],
                row['next_sentence'] if pd.notna(row['next_sentence']) else ''
            ])).strip(),  # Added strip() to remove extra spaces
            axis=1
        )
        
        print(f"Sampled {len(df)} rows with context")
    except Exception as e:
        print(f"Error creating final DataFrame: {e}")
        sys.exit(1)

    # take out the 2% longest and shortest paragraphs
    df['lenght'] = df['paragraph'].str.len()
    # Calculate the 2nd and 98th percentiles of paragraph length
    lower_bound = df['lenght'].quantile(0.02)
    upper_bound = df['lenght'].quantile(0.98)

    # Filter out paragraphs that are too short or too long
    df = df[(df['lenght'] > lower_bound) & (df['lenght'] < upper_bound)]
    print(f"After filtering by length: {len(df)} rows remaining")

    # Construct the dataset field names based on provided language codes.
    flores_source_field = f"sentence_{args.source_lang}"
    flores_target_field = f"sentence_{args.target_lang}"

    try:
        # Build few-shot examples from the Flores dev split.
        fewshot_examples = []
        dev_data = load_dataset("facebook/flores", f"{args.source_lang}-{args.target_lang}", split="dev")

        for example in dev_data.select(range(args.fewshot_num)):
            fewshot_examples.append({
                "source": example[flores_source_field],
                "target": example[flores_target_field],
            })
        fewshot = build_fewshot_prompt(fewshot_examples, args.source_label, args.target_label)
        print(f"Built few-shot prompt with {args.fewshot_num} examples")
        
        #add the few shot prompt to the dataset
        df['prompt'] = df.paragraph.map(lambda x: f"{fewshot}{args.source_label}: {x}\n{args.target_label}: ")
        
        # At the end, use args.output_file if provided
        output_file = args.output_file if args.output_file else f'{df.shape[0]}_clean_wiki.jsonl'
        df[['paragraph', 'prompt']].to_json(output_file, lines=True, orient='records')
        print(f"Saved {len(df)} samples to {output_file}")
    except Exception as e:
        print(f"Error creating prompts or saving output: {e}")
        sys.exit(1)