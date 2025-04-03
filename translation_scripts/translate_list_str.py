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
from iso639 import Lang

tqdm.pandas() 

print('everything imported')

"""
This script translates an the ArenaHard from English to a target language.
It assumes that the text to be translated is in the column prompt.
It checks whether an Opus model exist for the English -> target language and if not:
use NLLB instead. Or you can set the --deepl or gemini flag and use DeepL/Gemini for translation.
Author: Maria
"""

def create_translation_prompt(source_sentence:str, target_lang:str, source_lang='English') -> str:
    prompt = f"""Translate the following {source_lang} sentence into a faithful and fluent {target_lang} translation. Do not include the source sentence in the response. Do not say anything else than the translation""
    Source sentence: {source_sentence}
    """
    return prompt

def call_litellm(prompt, model="gpt-4o", max_tokens=1000, temperature=0.0):
    try:
        response = completion(
            model=model,
            messages = [
                {"role": "system", "content": "You are a helpful text edit and translation assistant."},
                { "content": prompt, "role": "user"}
                ],
            stream=False,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        return response
    except Exception as e:
        sys.exit(e)

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
    parser.add_argument('--translate-col', nargs='+', help='The name of the column to translate', required=True)
    parser.add_argument('--src_lang', type=str, default='en', help='The source language, required=True')
    args = parser.parse_args()

    print("Translating colums", args.translate_col)
    
    src_lang = 'en'
    src_lang_name = Lang(src_lang).name
    tgt_lang_name = Lang(args.tgt_lang).name

    # Load the model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{args.tgt_lang}'

    output_file = os.path.join(args.output_dir, f"{args.tgt_lang}.jsonl")

    print(f"Translating MT bench questions from {src_lang_name} to {tgt_lang_name}")
    print(f"Reading en questions from {args.source_file}")

    # Check if the output directory exists, if not create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    #open the English source file depending on the format
    if args.source_file.split('.')[-1] == 'parquet': 
        df = pd.read_parquet(args.source_file)
    elif args.source_file.split('.')[-1] == 'jsonl': 
        df = pd.read_json(args.source_file, lines=True)
    elif args.source_file.split('.')[-1] == 'csv':
        df = pd.read_csv(args.source_file, sep=',')
    elif args.source_file.split('.')[-1] == 'tsv':
        df = pd.read_csv(args.source_file, sep='\t')
    else:
        print('Currently only supporting jsonl, tsv, csv, tsv and parquet')
        sys.exit(1)

    if args.max_samples:
        df = df.head(args.max_samples)

    # check if the opus model exists
    if opus_model_exists(model_name):
        print("Opus model found")
        using_opus = True
    else:
        print('Translating using GPT-4o')
        using_opus = False

    # Translate using Opus, translate each list item in case it is a list
    if using_opus:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        for col in args.translate_col:
            df.loc[:, f'{col}_mt-translated'] = df[col].progress_map(
            lambda x: [translate_opus(item, tokenizer=tokenizer, model=model) for item in x] 
                    if isinstance(x, list) 
                    else translate_opus(x, tokenizer=tokenizer, model=model)
            )

    else:
        for col in args.translate_col:
            df.loc[:, f'{col}_mt-translated'] = df[col].progress_map(
            lambda x: [call_litellm(create_translation_prompt(source_sentence=item, target_lang=src_lang_name)) for item in x] 
                    if isinstance(x, list) 
                    else call_litellm(create_translation_prompt(source_sentence=x, target_lang=tgt_lang_name))
        )

                                                      
    #rename the english question column
    df.rename(columns={col: f'{col}_{src_lang}'}, inplace=True)
    
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)

    print(f'Wrote to file {output_file}')
    print('Done')