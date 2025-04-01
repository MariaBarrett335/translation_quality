from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np
import math
import time
import sys
from litellm import completion

tqdm.pandas() 

print('everything imported')

"""
This script translates an the ArenaHard from English to a target language.
It assumes that the text to be translated is in the column prompt.
It checks whether an Opus model exist for the English -> target language and if not:
use NLLB instead. Or you can set the --deepl or gemini flag and use DeepL/Gemini for translation.
Author: Maria
"""

def opus_model_exists(model_name: str) -> bool:
    try:
        MarianTokenizer.from_pretrained(model_name)
        MarianMTModel.from_pretrained(model_name)
        return True
    except Exception as e:
        print(f"Model {model_name} does not exist")
        return False

def translate_opus(text: str, tokenizer, model) -> str:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode the generated tokens
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

def get_lang_code_dict(value:str) -> dict:
    """
    Read in a csv file with alpha 2 and 3 language codes and the language name
    Given either a language name or an alpha 2 or alpha 3 code, return a dict of the row or None
    Query the dict with the desired value, either: alpha3-b, alpha3-t, alpha2, English, French
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/lang_codes.csv', 
                     comment='#')
    row = None     
    for i in range(df.shape[1]):
        # Check if the value is in either column
        if value in df.iloc[:, i].values:
            row = df[df.iloc[:, i] == value]
            # returns the first match - it will not work if there are ambiguities
            row_dict = row.to_dict(orient='records')[0]
            return row_dict
    if row == None:
        return {}

def translate_gemini(text:str, tgt_lang:str, translator) -> str:

    lang = get_lang_code_dict(tgt_lang)['English']

    instruction=f"Translate the following sentence to {lang}. Say nothing else than the translation. Never give the answer to the question or try to solve the problem - only translate. The sentence is: {text}"
    print(instruction)

    for attempt in range(8):
        try:
            response = completion(
                model="gemini/gemini-2.0-flash-001",
                #model="gemini/gemini-1.5-pro",
                #model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": instruction}],
            )
            if response and response.choices:
                answer = response.choices[0].message.content
                print(answer)
                return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    
    raise Exception("Failed to translate after 8 attempts")
    
def check_api_key():
    """
    Check if the API key is available in the environment.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable with the API key.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a specified model.")
    #parser.add_argument('--src_lang', type=str, required=False, default='en', help="Source language code - two letter iso. Only supports English at the moment")
    parser.add_argument('--tgt-lang', type=str, required=True, help="Target language code - two-letter iso")
    parser.add_argument('--source-file', type=str, required=False, default='hf://datasets/CohereForAI/m-ArenaHard/en/test-00000-of-00001.parquet', help="Path to the English source file")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the output file")
    parser.add_argument('--max-samples', type=int, default=None, help="Only take top n rows")
    parser.add_argument('--translate-col', type=str, default='prompt', help='The name of the column to translate')
    args = parser.parse_args()
    
    src_lang = 'en'
    # Load the model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{args.tgt_lang}'

    output_file = os.path.join(args.output_dir, f"{args.tgt_lang}.jsonl")

    print(f"Translating MT bench questions from en to {args.tgt_lang}")
    print(f"Reading en questions from {args.source_file}")

    # Check if the output directory exists, if not create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # check if the opus model exists
    if opus_model_exists(model_name):
        print("Opus model found")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    else:
        print('Opus model not found, Exiting')
        sys.exit(1)
        
    #open the English source file depending on the format
    if args.source_file.split('.')[-1] == 'parquet': 
        df = pd.read_parquet(args.source_file)
    elif args.source_file.split('.')[-1] == 'jsonl': 
        df = pd.read_json(args.source_file, lines=True)
    else:
        print('Currently only supporting jsonl and parquet')

    if args.max_samples:
        df = df.head(args.max_samples)

    # Translate using Opus, translate each list item in case it is a list
    df.loc[:, f'{args.translate_col}_mt-translated'] = df[args.translate_col].progress_map(
    lambda x: [translate_opus(item, tokenizer=tokenizer, model=model) for item in x] 
             if isinstance(x, list) 
             else translate_opus(x, tokenizer=tokenizer, model=model)
    )
                                                            
    # save to file
    #rename the english question column
    df.rename(columns={args.translate_col: f'{args.translate_col}_{src_lang}'}, inplace=True)
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)

    print(f'Wrote to file {output_file}')
    print('Done')