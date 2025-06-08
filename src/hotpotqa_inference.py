import json
import os
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from traverse_algo import Hotpot_traverse, retrieve_traverse
from utils import chat_with_gpt4, extract_answer
from BM25 import BM_find_document
from collections import Counter


def make_prompt(question, documents, baseline=False):
    if baseline:
        prompt = open("../prompt/hotpot_COT.txt", "r").read()
    else:
        prompt = open("../prompt/hottop_prompt.txt", "r").read()
    input_q = question#+ "\n" + "subquestion:\n"+str(questions)
    prompt = prompt.replace("{Question}",input_q )
    documents = ["#context:\n " + doc for doc in documents]
    documents_str = "\n".join(documents)
    prompt = prompt.replace("{Documents}", documents_str)
    return prompt
    
class HotpotProcessor:
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the hotpotqa processor.
        Args:
            model_name: The name of the language model to use for processing.
        """
        self.traverse =retrieve_traverse() #Hotpot_traverse()
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0

    def retrieve_answer(self, question: str) -> Tuple[List[str], str, str]:
        """Retrieve and process answers for a given question.
        
        Args:
            question: The input question to process.
            
        Returns:
            A tuple containing:
            - List of relevant documents
            - List of disambiguated questions
            - Combined response
            - Dictionary mapping questions to their individual answers
        """
        try:
            questions, documents = self.traverse.disambiguate_text(text=question)
            all_docs =documents
            prompt = make_prompt(question, all_docs )
            response,inp,out = chat_with_gpt4(prompt)
            self.input_tokens += inp
            self.output_tokens += out
            return all_docs, response, extract_answer(response)
        except Exception as e:
            print(e)
            documents = BM_find_document(question, 15)
            prompt = make_prompt(question, documents)
            response,inp,out = chat_with_gpt4(prompt)
            self.input_tokens += inp
            self.output_tokens += out
            return documents, response, extract_answer(response)
    def baseline(self, question) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        times = 10
        responses = []
        simple_documents = BM_find_document(question,15)#self.traverse.disambiguate_text(question)
        simple_prompt = make_prompt(question,[], True)
        for i in range(times):
            a,inp,out = chat_with_gpt4(simple_prompt)
            self.input_tokens += inp
            self.output_tokens += out
            try:
                responses.append(extract_answer(a))
            except Exception as e:
                responses.append(a)
        counter = Counter(responses)
        top = counter.most_common(1)
        top_elements = [item[0] for item in top]
        return simple_documents, responses, top_elements[0]
    def process_dataset(self, input_path: str, output_path: str,processed_output:str, usage_path:str, max_samples: int = None) -> None:
        """Process the ASQA dataset and save results.
        
        Args:
            input_path: Path to the input ASQA dataset.
            output_path: Path to save the processed output.
            max_samples: Maximum number of samples to process (None for all).
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(input_path, 'r') as f:
            dataset = json.load(f)
        #with open(output_path, 'r') as f:
        #    processed_data = json.load(f)
        #with open(processed_output, 'r') as f:
        #    answer_format = json.load(f)
        processed_data = []
        answer_format = {}
        answer_format["answer"] = {}
        for idx, example in enumerate(tqdm(dataset)):
            if max_samples and idx >= max_samples:
                break
            if example['_id'] not in answer_format["answer"].keys():
                instance = {
                    'id':example['_id'],
                    'question': example['question'],
                    'gold': example['answer'],
                }
                documents, step_by_step, answer = self.baseline(
                    instance['question'])
                instance.update({
                    'generated_answer': answer,
                    'step_by_step': step_by_step,
                    'documents': documents
                })
                answer_format['answer'][example['_id']] = [answer, step_by_step]
                processed_data.append(instance)
                if idx % 10 == 0:
                    # Save processed data
                    with open(output_path, 'w') as f:
                        json.dump(processed_data, f, indent=2)
                    with open(processed_output, 'w') as f:
                        json.dump(answer_format, f, indent=2)
        # Save processed data
        print(usage_path)
        with open(usage_path,'w') as f:
            self.input_tokens += self.traverse.input_tokens
            self.output_tokens += self.traverse.output_tokens
            obj = {"input_data":self.input_tokens, "output_data":self.output_tokens}
            json.dump(obj,f,indent=2)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        with open(processed_output, 'w') as f:
            json.dump(answer_format, f, indent=2)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process ASQA dataset and convert to QA format.')
    
    parser.add_argument('--input_dir', required=True,
                      help='Directory containing the input ASQA dataset')
    parser.add_argument('--output_dir', required=True,
                      help='Directory to save the processed output')
    parser.add_argument('--max_samples', type=int, default=50,
                      help='Maximum number of samples to process (default: process all)')
    parser.add_argument('--model_name', default='gpt-4o-mini',
                      help='Name of the language model to use (default: gpt-4)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize processor
    processor = HotpotProcessor(model_name=args.model_name)
    
    # Set up input and output paths
    input_path = args.input_dir
    processed_output_path = os.path.join(args.output_dir, 'hotpotqa.json')
    formated_output_path = os.path.join(args.output_dir, 'formated_hotpotqa.json')   
    usage_path = os.path.join(args.output_dir,'usage.json')   
    # Process ASQA dataset
    print(f"Processing dataset from {input_path}")
    processor.process_dataset(input_path, processed_output_path,formated_output_path,usage_path, max_samples=args.max_samples)
    print(f"Processed data saved to {processed_output_path}")

if __name__ == "__main__":
    main()
             