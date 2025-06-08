import json
import os
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from traverse_algo import AmbigDoc_traverse, retrieve_traverse
from utils import chat_with_gpt4, extract_answer, usc
from BM25 import BM_find_document
from collections import Counter
def make_prompt(question, documents, baseline=False):
    if baseline:
        prompt = open("../dev/prompt/Ambig_Doc_COT.txt", "r").read()
    else:
        prompt = open("../dev/prompt/Ambig_Doc.txt", "r").read()
    input_q = question#+ "\n" + "subquestion:\n"+str(questions)
    prompt = prompt.replace("{Question}",input_q )
    documents = ["#context:\n " + doc for doc in documents]
    documents_str = "\n".join(documents)
    prompt = prompt.replace("{Documents}", documents_str)
    return prompt
class AmbigDocprocessor:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.traverse = retrieve_traverse()
        self.input_tokens = 0
        self.output_tokens = 0


    def retrieve_answer(self, question: str) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        questions, documents = self.traverse.disambiguate_text(question)
        all_docs =documents
        #print(documents)
        prompt = make_prompt(question, all_docs )
        step_by_step, i , o= chat_with_gpt4(prompt)
        self.input_tokens += i
        self.output_tokens += o
        try:
            combined_response = extract_answer(step_by_step)
        except Exception as e:
            print(e)
            questions = [question]
            documents = BM_find_document(question, 15)
            all_docs = documents
            prompt = make_prompt(question, documents)
            response, i ,o = chat_with_gpt4(prompt)
            combined_response = extract_answer(response) 
            self.input_tokens += i
            self.output_tokens += o    
        return all_docs, questions, step_by_step, combined_response
            
    def baseline(self, question) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        times = 10
        responses = []
        simple_documents = BM_find_document(question,15)
        simple_prompt = make_prompt(question,simple_documents, True)
        for i in range(times):
            a, i , o = chat_with_gpt4(simple_prompt)
            r = extract_answer(a)
            responses.append(r)
            self.input_tokens += i
            self.output_tokens += o 
        r,i,o = usc(question, responses)
        self.input_tokens += i
        self.output_tokens += o 
        return simple_documents,question, a, r
    def process_dataset(self, input_path: str, output_path: str,usage_path:str, max_samples: int = None) -> None:
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
        processed_data = []
        #with open(output_path, 'r') as f:
        #    processed_data = json.load(f)
        len_p = len(processed_data)
        for i in range(0,500):
            if i > max_samples: break
            example = dataset[i]
            instance = example
            if i >= len_p:
                # Process the ambiguous question
                documents, questions, step_by_step, answer = self.baseline(
                    instance['question']
                )

                # Add processed information
                instance.update({
                    'gen_answer': answer,
                    'step_by_step': step_by_step,
                    'questions': questions,
                    'retrieve_documents': documents
                })

                processed_data.append(instance)
                if idx % 20 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(processed_data, f, indent=2)
        # Save processed data
        with open(usage_path,'w') as f:
            self.input_tokens += self.traverse.input_tokens
            self.output_tokens += self.traverse.output_tokens
            obj = {"input_data":self.input_tokens, "output_data":self.output_tokens}
            json.dump(obj,f,indent=2)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)

    def convert_to_qa_format(self, predictions: List[Dict], output_dir: str) -> None:
        """Convert processed data to QA format for evaluation.
        
        Args:
            predictions: List of processed examples.
            output_dir: Directory to save the converted format.
        """
        res = {'data': []}
        
        for ex in predictions:
            question, gen_ans = ex["question"], ex["gen_answer"]
            gen_ans = ' '.join(gen_ans.replace("\n", " ").split())
            idx = question.lower().find(ex["ambiguous_entity"])
            used_ids = range(min(len(ex["documents"]), 5))

            for i, uid in enumerate(used_ids):
                question_sub = question
                if idx != -1: question_sub = question[:idx]+ex["documents"][uid]["title"]+question[idx+len(ex["ambiguous_entity"]):]
                #print(ex["documents"])
                res['data'].append({
                    'context': gen_ans,
                    'id': str(ex["qid"])+"_"+str(i),
                    'question': question_sub,
                    'answers': {'text': [ex["documents"][uid]["answer"]], 'answer_start': []}
                })
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'formatted_qa.json')
        with open(output_path, 'w') as f:
            json.dump(res, f, indent=2)

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
    processor = AmbigDocprocessor(model_name=args.model_name)
    
    # Set up input and output paths
    input_path = args.input_dir
    processed_output_path = os.path.join(args.output_dir, 'processed_ambigqa.json')
    qa_format_dir = os.path.join(args.output_dir, 'qa_format')
    usage_path = os.path.join(args.output_dir, "usage.json")
    # Process ASQA dataset
    print(f"Processing dataset from {input_path}")
    processor.process_dataset(input_path, processed_output_path,usage_path, max_samples=args.max_samples)
    print(f"Processed data saved to {processed_output_path}")
    
    # Convert to QA format
    print("Converting to QA format...")
    with open(processed_output_path, 'r') as f:
        processed_data = json.load(f)
    processor.convert_to_qa_format(processed_data, qa_format_dir)
    print(f"QA format data saved to {qa_format_dir}")

if __name__ == "__main__":
    main()