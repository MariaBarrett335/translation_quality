import pandas as pd
import sys
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
    
    parser.add_argument('--lang', type=str,
                        default="en",
                        help='two-letter iso language code')
    
    parser.add_argument('--output_file', type=str,
                        help='Path to output CSV file (default: [input_filename]_rated.csv)')
    
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum completion tokens (default: 1000)')
    parser.add_argument('--temp', type=float, default=0.1, help="The temperature setting. Default 0.1" )
    args = parser.parse_args()
    return args

def create_prompt(question=str, cot=False):
    if cot:
        prompt = f"""Answer the following question and give the reason for this answer. Follow this format:
        <reason>Give the reason for the answer</reason> 
        <answer>Write the answer to the question </answer>
        <question>{question}</question> """
        return prompt
    else: 
        return question
    

def get_dataset_section(lang:str)->pd.DataFrame:
    #takes a two-letter language code and returns the section of the Eloquent Robustness dataset as a dataframe
    data = load_dataset("Eloquent/Robustness", "test")
    languages = data['test'][0]['eloquent-robustness-test']['languages']
    selected_section = next((lang_data for lang_data in languages if lang_data['language'] == lang), None)

    if not selected_section:
        print(f"{lang} section not found")
        return None
    else:
        df = pd.DataFrame.from_dict(selected_section['items'])
        return(df)


def main(args=None, pre_loaded_model=None, pre_loaded_tokenizer=None):
    # Parse command-line arguments
    args = args or parse_arguments()
        
    # Display configuration
    print(f"Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Chain of thought: {'Enabled' if args.cot else 'Disabled'}")
    print(f"  - Max tokens: {args.max_tokens}")
    print(f"  - Output file: {args.output_file}")
    
    # Load the dataset
    print(f"Loading {args.lang} dataset from Eloquent Robustness...")
    df = get_dataset_section(args.lang)

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

    df['answer'] = df['prompt'].progress_map(lambda x: get_response(prompt=x, model=model, tokenizer=tokenizer, max_completion_tokens=args.max_tokens, temperature=args.temp))
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