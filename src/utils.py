import openai
import json
from pathlib import Path
import pandas as pd
import re
import string
from rerankers import Reranker
from nltk.corpus import wordnet as wn
import os
ranker = Reranker("ms-marco-MiniLM-L-12-v2", model_type='flashrank')

api_key = os.environ.get('OPENAI_API_KEY')
model = os.environ.get('MODEL')
url = os.environ.get("URL")

client = openai.Client(api_key = api_key, base_url = url)
import wikipediaapi
from typing import List
from PyDictionary import PyDictionary
from googlesearch import search

# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
def chat_with_gpt4(message):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
        )
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        return completion.choices[0].message.content, input_tokens, output_tokens
    except Exception as e:
        print(f"Error status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        print(f"Error message: {str(e)}")
        return None

def rerank_document(input: List[str],q:str, num=30):
    if len(input) <= num:
        return input
    b = set()
    for k in input:
        b.add(k)
    results = ranker.rank(query=q,docs=list(b))
    a = set()
    for r in results.top_k(len(b)):
        a.add(r.text)
        if len(a) >= num:
            break
    return list(a)
def google_search(input:str):
    result = search(input, advanced=True)
    output = ""
    for i in result:
        output += i.description + "\n"
    return output
def extract_answer(text) -> str:
    answer_start = text.find("#FINAL")    
    if answer_start == -1:
        return ""
    answer = text[answer_start + len("#FINAL"):]
    return answer.strip()
    
def parquet_to_json(parquet_file, output_file=None, orient='records'):
    """
    Convert a Parquet file to JSON format.
    
    Parameters:/Users/bryanzhang/Downloads/Asqa main data.parquet
    parquet_file (str): Path to input Parquet file
    output_file (str, optional): Path to output JSON file. If None, returns JSON string
    orient (str, optional): Orient format for JSON output. Default is 'records'
        Possible values:
        - 'records': list-like [{column -> value}, ... ]
        - 'split': dict-like {index -> [index], columns -> [columns], data -> [values]}
        - 'index': dict-like {index -> {column -> value}}
        - 'columns': dict-like {column -> {index -> value}}
        - 'values': just the values array
        - 'table': dict-like {'schema': {schema}, 'data': {data}}
    
    Returns:
    str or None: If output_file is None, returns JSON string. Otherwise writes to file and returns None.
    """
    try:
        # Read Parquet file
        df = pd.read_parquet(parquet_file)
        
        # Convert to JSON
        if output_file:
            # Write directly to file
            df.to_json(output_file, orient=orient, indent=2)
            return None
        else:
            # Return JSON string
            return df.to_json(orient=orient, indent=2)
            
    except Exception as e:
        raise Exception(f"Error converting Parquet to JSON: {str(e)}")


def read_jsonl(file_path: str):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON on line: {e.lineno}", e.doc, e.pos)
      
def write_to_json(data, 
                  file_path: str,
                  indent: int = 4,
                  ensure_ascii: bool = False) -> None:
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Validate input type
    if not isinstance(data, (dict, list)):
        raise TypeError("Data must be a dictionary or a list of dictionaries")
    
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise TypeError("All items in the list must be dictionaries")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, 
                     indent=indent, 
                     ensure_ascii=ensure_ascii,
                     sort_keys=False)
    except OSError as e:
        raise OSError(f"Error writing to file {file_path}: {str(e)}")
    
def make_prompt(question, documents):
    prompt = open("../prompt/prompt.txt", "r").read()
    prompt = prompt.replace("{Question}", question)
    documents = ["#context:\n " + doc for doc in documents]
    documents_str = "\n".join(documents)
    insertion_point = prompt.find("let's think step by step")
    if insertion_point == -1:
        raise ValueError("The prompt must contain the phrase 'let's think step by step'")
    updated_prompt = prompt[:insertion_point] + documents_str + "\n\n" + prompt[insertion_point:]
    return updated_prompt
def make_baseline(question, documents):
    prompt = open("../prompt/baseline.txt","r").read()
    prompt = prompt.replace("{Question}", question)
    documents = ["#context:\n " + doc for doc in documents]
    documents_str = "\n".join(documents)
    prompt += documents_str
    return prompt
def make_prompt_asqa(question, documents):
    prompt = open("../prompt/asqa_prompt.txt", "r").read()
    prompt = prompt.replace("{Question}", question)
    documents = [f"{i+1}:\n " + doc for i, doc in enumerate(documents)]
    documents_str ="#context" +  "\n".join(documents)
    insertion_point = prompt.find("let's think step by step")
    if insertion_point == -1:
        raise ValueError("The prompt must contain the phrase 'let's think step by step'")
    updated_prompt = prompt[:insertion_point] + documents_str + "\n\n" + prompt[insertion_point:]
    return updated_prompt


def get_all_definitions(word_or_phrase):
    # Split phrase into words
    words = word_or_phrase.strip().split()
    
    if len(words) == 1:
        # Single word case
        synsets = wn.synsets(word_or_phrase)
        definitions = {word_or_phrase:[]}
        
        for i, syn in enumerate(synsets, 1):
            pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 
                      's': 'adjective satellite', 'r': 'adverb'}
            pos = pos_map.get(syn.pos(), syn.pos())
            
            definition = {
                'part_of_speech': pos,
                'definition': syn.definition(),
            }
            definitions[word_or_phrase].append(definition)
            
        return definitions
    else:
        # Handle multi-word phrase
        # First try exact match (some phrases are in WordNet)
        compound_synsets = wn.synsets('_'.join(words))
        if compound_synsets:
            return get_all_definitions('_'.join(words))
        
        # If no exact match, get definitions for each word
        phrase_definitions = {}
        for word in words:
            phrase_definitions[word] = get_all_definitions(word)
        
        return phrase_definitions

"""chat_with_gpt4("hello")"""

def usc(question:str, responses: List[str]) ->str:
    prompt = f"I have generated the following responses to the question: {question}"
    response_str = ""
    for i in range(len(responses)):
        response_str += f"response {i}: {responses[i]}\n"
    prompt = f"I have generated the following responses to the question: {question}\n{response_str}\n Evaluate these responses. Select the most consistent response based on majority consensus. Directly give the most consistent response"
    return chat_with_gpt4(prompt)
    