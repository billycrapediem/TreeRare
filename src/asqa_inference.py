import json
import os
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from traverse_algo import ASQA_traverse, retrieve_traverse
from utils import make_prompt_asqa, chat_with_gpt4, extract_answer, make_baseline, usc
from BM25 import BM_find_document
from collections import Counter
class ASQAProcessor:
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the ASQA processor.
        
        Args:
            model_name: The name of the language model to use for processing.
        """
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_name = model_name
        self.traverse = retrieve_traverse()

    def retrieve_answer(self, question: str) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        answers_dict = {}
        # Try disambiguation approach first
        questions, documents = self.traverse.disambiguate_text(question)# Process each disambiguated question
        questions.append(question)
        for q in questions:
            tmp = []
            prompt = make_prompt_asqa(q, documents + tmp )
            r, i, o = chat_with_gpt4(prompt)
            answers_dict[q] = r
            self.input_tokens += i
            self.output_tokens += o
        # Combine individual answers into a coherent response
        r, i , o = self._combine_answers(answers_dict) 
        combined_response = r
        self.input_tokens += i
        self.output_tokens += o
        return documents, questions, combined_response, answers_dict

    def _combine_answers(self, answers_dict: Dict[str, str]) -> str:
        """Combine multiple answers into a coherent paragraph.
        
        Args:
            answers_dict: Dictionary mapping questions to their answers.
            
        Returns:
            A combined response paragraph.
        """
        prompt = "combine all of them into long-form paragraph. (don't summerize information)"
        
        for key, answer in answers_dict.items():
            prompt += f"Question: {key} Answer: {answer}\n"
            
        return chat_with_gpt4(prompt)
    def baseline(self, question) -> Tuple[List[str], List[str], str, Dict[str, str]]:
        times = 10
        responses = []
        simple_documents = BM_find_document(question,15)
        simple_prompt = make_baseline(question,simple_documents)
        try:
            for i in range(times):
                ans,inp,out = chat_with_gpt4(simple_prompt)
                self.input_tokens += inp
                self.output_tokens += out
                responses.append(extract_answer(ans))
        except Exception as e:
            print(f"error: {e}")
        ans,i,o = usc(question=question, responses=responses)
        self.input_tokens += i
        self.output_tokens += o
        return simple_documents, [question], ans, responses
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
        for idx, example in enumerate(tqdm(dataset)):
            if max_samples and idx >= max_samples:
                break
            instance = {
                'qa_pairs': example['qa_pairs'],
                'ambiguous_question': example['ambiguous_question'],
                'sample_id': example['sample_id']
            }
            # Process the ambiguous question
            documents, questions, answer, step_by_step = self.baseline(
                instance['ambiguous_question']
            )
            
            # Add processed information
            instance.update({
                'generated_answer': answer,
                'step_by_step': step_by_step,
                'questions': questions,
                'documents': documents
            })
            
            processed_data.append(instance)
            if idx % 20 == 0:
            # Save processed data
                with open(output_path, 'w') as f:
                        json.dump(processed_data, f, indent=2)
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
        qa_data = {'data': []}
        
        for example in predictions:
            context = example['generated_answer']
            
            for idx, qa_pair in enumerate(example['qa_pairs']):
                qa_data['data'].append({
                    'context': context,
                    'id': f"{example['sample_id']}_{idx}",
                    'question': qa_pair['question'],
                    'answers': {
                        'text': qa_pair['short_answers'],
                        'answer_start': []
                    }
                })
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'qa.json')
        
        with open(output_path, 'w') as f:
            json.dump(qa_data, f, indent=2)

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
    processor = ASQAProcessor(model_name=args.model_name)
    
    # Set up input and output paths
    input_path = os.path.join(args.input_dir, 'asqa.json')
    processed_output_path = os.path.join(args.output_dir, 'processed_asqa.json')
    qa_format_dir = os.path.join(args.output_dir, 'qa_format')
    usage_path = os.path.join(args.output_dir,"usage.json")
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