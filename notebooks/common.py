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

def create_rating_prompt(candidate_A:str, candidate_B:str, cot=False):

    prompt = f"""Here are two sentences. One is more fluent than the other. Select the candidate that appears more fluent to a native Danish speaker. Do not focus on the helpfulness or the content of the text, only its use of language.
    Return only the letter of the candidate, i.e. A or B. Say nothing else.
    A: {candidate_A}
    B: {candidate_B}
    """

    prompt_cot = f"""Here are two sentences. One is more fluent than the other. Select the candidate that appears more fluent to a native Danish speaker. Do not focus on the helpfulness or the content of the text, only its use of language.
    Return the fluency evaluation of each candidate between these tags <candidate_a_fluency_rating> </candidate_a_fluency_rating> and <candidate_b_fluency_rating> </candidate_b_fluency_rating>. Return the letter of the most fluent candidate, i.e. A or B between these tags: <winner></winner>.
    In case of the same number of points, select the candidate with the less severe errors as the winner.
    The fluency rating should be according to this rubric:
    
    1 Point: Minimal Fluency

    Grammar: Technically grammatical but with pervasive errors in most areas (gender, number, tense, etc.)
    Vocabulary: Very basic vocabulary with significant repetition and many direct translations
    Pronunciation/Flow: Text is choppy and disconnected, would sound extremely unnatural to native speakers
    Sentence Structure: Almost exclusively simple or fragmented sentences arranged in an unnatural sequence
    Idioms: No awareness of Danish cultural context in language use

    2 Points: Basic Fluency

    Grammar: Frequent grammatical errors in article use, and verb tenses, though main meaning is understandable
    Vocabulary: Limited vocabulary with repetition and occasional use of non-Danish words or direct translations
    Pronunciation/Flow: Text reads with a distinctly non-Danish cadence and would sound unnatural when read aloud
    Sentence Structure: Predominantly simple sentences with awkward attempts at complexity
    Idioms: Minimal awareness of Danish idioms and verbal phrases

    3 Points: Intermediate Fluency

    Grammar: Some noticeable grammatical errors, particularly with complex structures, but meaning remains clear
    Vocabulary: Adequate vocabulary for most situations, but limited idiomatic expressions and some repetition
    Pronunciation/Flow: Text has a somewhat unnatural rhythm that would be noticeable to native speakers
    Sentence Structure: Mix of simple and complex sentences, but reliance on certain patterns
    Idioms: Some awareness of Danish idioms and verbal phrases, but still some errors that are directly translated from English, e.g., 'jeg bryder problemet ned'

    4 Points: Advanced Fluency

    Grammar: Very few minor grammatical errors that wouldn't distract a native speaker
    Vocabulary: Broad vocabulary with good use of idioms, though occasional imprecise word choice
    Pronunciation/Flow: Text flows naturally with only occasional awkward phrasing
    Sentence Structure: Good variety of complex sentence structures with minor awkwardness
    Idioms: Generally appropriate use of Danish idioms and verbal phrases with occasional slight misuse

    5 Points: Native-Like Fluency

    Grammar: Perfect grammatical control with no errors in noun gender, verb conjugation, or word order
    Vocabulary: Rich, precise, and idiomatic vocabulary with proper use of Danish expressions and colloquialisms
    Flow: Text has a natural rhythm that would sound completely authentic when read aloud
    Sentence Structure: Varied and complex sentence structures used appropriately and effortlessly
    Idioms: Appropriate use of Danish idioms, verbal phraess, and Danish-specific expressions.
    
    A: {candidate_A}
    B: {candidate_B}
    """

    if cot ==False:
        return prompt
    else: 
        return prompt_cot

def create_minimal_rating_prompt(candidate_A:str, candidate_B:str):
    prompt = f"""A: {candidate_A}
    B: {candidate_B}
    """
    return prompt

def create_batch_prompt(dict_of_prompts: dict, cot=False):
    # Create a properly formatted string representation of the dictionary
    dict_as_string = str({k: v for k, v in dict_of_prompts.items()})
    
    prompt = f"""Below you will see a dictionary of indices and prompts. For each index, select the candidate that appears more fluent to a native Danish speaker. Do not focus on the helpfulness, or the content of the text, only its use of language.
    Punish the candidate that contains anglicisms, ungrammaticality, translationese, or other non-native language features.
    For each index, return only the letter of the candidate, i.e. A or B. Say nothing else.
    Return a dictionary like this: {{0: 'A', 1: 'B', 2: 'A', ...}}
    {dict_as_string}
    """

    prompt_cot = f"""Below you will see a dictionary of indices and prompts. For each index, select the candidate that appears more fluent to a native Danish speaker. Do not focus on the helpfulness, or the content of the text, only its use of language.
    Punish the candidate that contains anglicisms, ungrammaticality, translationese, or other non-native language features.
    For each index, return the letter of the candidate, i.e. A or B, as well as the reason for the choice.
    Return a dictionary like this:
    {{
    0: {{'winner': 'A', 'reason': 'Candidate B contains an anglicism in the form of non standard verb-particle construction ('tjekke ned')'}},
    1: {{'winner': 'B', 'reason': 'Candidate A contains a non-standard Danish word: (radioactive)'}},
    ...
    }}
    {dict_as_string}
    """
    if cot:
        return prompt_cot
    else:
        return prompt

def create_prompt_rating(text:str):
    prompt=f"""Grade the following sentence according to this grading rubric. 
    <sentence>{text}</sentence>
    
    **Criterion **
    Danish Language Fluency Grading Rubric

    1 Point: Minimal Fluency

    Grammar: Technically grammatical but with pervasive errors in most areas (gender, number, tense, etc.)
    Vocabulary: Very basic vocabulary with significant repetition and many direct translations
    Pronunciation/Flow: Text is choppy and disconnected, would sound extremely unnatural to native speakers
    Sentence Structure: Almost exclusively simple or fragmented sentences arranged in an unnatural sequence
    Idioms: No awareness of Danish cultural context in language use

    2 Points: Basic Fluency

    Grammar: Frequent grammatical errors in article use, and verb tenses, though main meaning is understandable
    Vocabulary: Limited vocabulary with repetition and occasional use of non-Danish words or direct translations
    Pronunciation/Flow: Text reads with a distinctly non-Danish cadence and would sound unnatural when read aloud
    Sentence Structure: Predominantly simple sentences with awkward attempts at complexity
    Idioms: Minimal awareness of Danish idioms and verbal phrases

    3 Points: Intermediate Fluency

    Grammar: Some noticeable grammatical errors, particularly with complex structures, but meaning remains clear
    Vocabulary: Adequate vocabulary for most situations, but limited idiomatic expressions and some repetition
    Pronunciation/Flow: Text has a somewhat unnatural rhythm that would be noticeable to native speakers
    Sentence Structure: Mix of simple and complex sentences, but reliance on certain patterns
    Idioms: Some awareness of Danish idioms and verbal phrases, but still some errors that are directly translated from English, e.g., 'jeg bryder problemet ned'

    4 Points: Advanced Fluency

    Grammar: Very few minor grammatical errors that wouldn't distract a native speaker
    Vocabulary: Broad vocabulary with good use of idioms, though occasional imprecise word choice
    Pronunciation/Flow: Text flows naturally with only occasional awkward phrasing
    Sentence Structure: Good variety of complex sentence structures with minor awkwardness
    Idioms: Generally appropriate use of Danish idioms and verbal phrases with occasional slight misuse

    5 Points: Native-Like Fluency

    Grammar: Perfect grammatical control with no errors in noun gender, verb conjugation, or word order
    Vocabulary: Rich, precise, and idiomatic vocabulary with proper use of Danish expressions and colloquialisms
    Flow: Text has a natural rhythm that would sound completely authentic when read aloud
    Sentence Structure: Varied and complex sentence structures used appropriately and effortlessly
    Idioms: Appropriate use of Danish idioms, verbal phraess, and Danish-specific expressions.

    ** Format 
    List all errors between the tags <reason> </reason>. In case of no fluency errors, write 'None' between the tags.
    Give your score (1-5) between the tags <score> </score>
    """

    return prompt