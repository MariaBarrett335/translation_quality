import pandas as pd
from tqdm import tqdm
import time
import litellm
import os
import numpy as np
tqdm.pandas()
#from .autonotebook import tqdm as notebook_tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import random
random.seed(42)
from evaluate import load
bertscore = load("bertscore")
chrf = load("chrf")

from litellm import completion
import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from bs4 import BeautifulSoup
import json
from collections import Counter
from scipy.stats import rankdata

def scores_to_ranks_dict(scores):
    """
    Convert a list of scores to a dictionary mapping scores to ranks.
    
    Higher scores get lower ranks (1 is best).
    Equal scores get the same rank.
    
    Args:
        scores: List of numerical scores
        
    Returns:
        Dictionary mapping scores to their corresponding ranks
    """
    unique_scores = sorted(set(scores), reverse=True)  # Sort in descending order
    rank_dict = {score: int(rank) for rank, score in enumerate(unique_scores, 1)}
    return rank_dict


def run_rating(translation:str, correction:str, rating_prompt_func, model='gpt-4o', cot=False, repeat=3, max_completion_tokens=200):
    trans = 0
    corr = 0
    for i in range(repeat):
        if 'gemini' in model:
            response = gemini_chat(rating_prompt_func(candidate_A=translation, candidate_B=correction, cot=cot), model=model, max_completion_tokens=max_completion_tokens)
        elif 'gpt' in model:
            response = call_litellm(rating_prompt_func(candidate_A=translation, candidate_B=correction, cot=cot), model=model, max_completion_tokens=max_completion_tokens)
        if cot and BeautifulSoup(response, 'html.parser').find('winner'):
            response = BeautifulSoup(response, 'html.parser').find('winner').text

        response = response.strip().lower()
        if response == 'a':
            trans += 1
        elif response == 'b':
            corr += 1
        elif response in(['equal', 'tie']):
            trans += 0.5
            corr += 0.5
        else:
            print(response)
        
        if 'gemini' in model:
             response = gemini_chat(rating_prompt_func(candidate_A=correction, candidate_B=translation, cot=cot), model=model, max_completion_tokens=max_completion_tokens)
        elif 'gpt' in model:
            response = call_litellm(rating_prompt_func(candidate_A=correction, candidate_B=translation, cot=cot), model=model, max_completion_tokens=max_completion_tokens)
        if cot and BeautifulSoup(response, 'html.parser').find('winner'):
            response = BeautifulSoup(response, 'html.parser').find('winner').text
            
        response = response.strip().lower()

        if response == 'a':
            corr += 1
        elif response == 'b':
            trans += 1
        elif response in(['equal', 'tie']):
            trans += 0.5
            corr += 0.5
        else:
            print(response)
    return(corr/(repeat*2))



def call_litellm(prompt, model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct", max_completion_tokens=200):
    try:
        response = completion(
            model=model,
            messages = [
                {"role": "system", "content": "You are a helpful text edit and translation assistant."},
                #{"role": "user", "content": "You will get an English source sentence and a translation to danish. Write a more fluent and more faithful translation. Say nothing else.\nSource sentence: Snow leopards are among the world’s most elusive creatures in the wild and it is hard to catch even one on camera, let alone four, with the sighting being celebrated as a success story for Pakistan’s conservation efforts.\nTranslation: Sneleoparder er blandt verdens mest undvigende vilde væsner, og det er svært at fange kun én på kamera, endsige fire, og denne observation bliver hyldet som en succes for Pakistans bevaringsbestræbelser."},
                #{"role": "assistant", "content": "Sneleoparder er blandt verdens mest sky dyr, og det er svært at fange bare en enkelt på kamera, og langt sjældnere fire, og denne observation bliver hyldet som en succes for Pakistans naturbevaringsindsatser."},
                #{"role": "user", "content": "You will get an English source sentence and a translation to danish. Write a more fluent and more faithful translation. Say nothing else.\nSource sentence: The implications go far beyond salary, too. Clubs that pay more than $5,000 were more likely to provide non-financial benefits – like health insurance, housing benefits, etc. – than those who do not.\nTranslation: Konsekvenserne rækker langt ud over løn. Klubber, der bidrager med mere end $5.000, er mere tilbøjelige til at tilbyde ikke-økonomiske fordele - såsom sygesikring, bolighjælp osv. – end dem der ikke gør."},
                #{"role": "assistant", "content": " Konsekvenserne rækker langt ud over løn. Klubber, der betaler mere end $5.000, er også mere tilbøjelige til at tilbyde ikke-økonomiske fordele - såsom sygesikring, bolighjælp osv. – end dem der ikke gør."},
                { "content": prompt, "role": "user"}
                ],
            stream=False,
            max_completion_tokens=max_completion_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        sys.exit(e)

def check_api_key():
  """
  Check if the API key is available in the environment.
  """
  api_key = os.environ.get('GEMINI_API_KEY')
  if not api_key:
    print("Please set the GEMINI_API_KEY environment variable with the API key.")
    sys.exit(1)

def create_minimal_rating_prompt(candidate_A:str, candidate_B:str):
    prompt = f"""A: {candidate_A}
    B: {candidate_B}
    """
    return prompt


def gemini_chat(prompt: str, model="gemini/gemini-2.0-flash", max_completion_tokens=200) -> str:
  """
  Generate a response from the model based on a given prompt.
  """
  response = completion(
    model=model, 
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_completion_tokens=max_completion_tokens
  )
  if response and response.choices:
    answer = response.choices[0].message.content
    return answer
  else:
    return "No response from the model"

check_api_key()

def add_rating_prompt_column(df, sentence_col, corrected_col):
    """
    Adds a prompt column with randomized A/B assignment
    and tracks which version was assigned to which position.
    
    Returns the modified dataframe with new columns:
    - 'prompt': The generated prompt
    - 'original_is_A': True if original sentence is A, False if B
    """
    
    # Create new columns
    prompts = []
    original_is_A = []
    
    # Process each row
    for _, row in df.iterrows():
        sentence = row[sentence_col]
        corrected = row[corrected_col]
        
        # Randomly determine which is A and which is B
        if random.random() < 0.5:
            candidate_A = sentence
            candidate_B = corrected
            original_is_A.append(True)
        else:
            candidate_A = corrected
            candidate_B = sentence
            original_is_A.append(False)
        
        # Generate the prompt
        prompt = create_minimal_rating_prompt(candidate_A, candidate_B)
        prompts.append(prompt)
    
    # Add new columns to dataframe
    df['prompt'] = prompts
    df['backtranslated_is_A'] = original_is_A
    
    return df