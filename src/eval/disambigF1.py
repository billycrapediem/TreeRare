import json
import re
import string
import collections
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from rouge_score import rouge_scorer
import sys
import nltk
from nltk.corpus import stopwords

class TextNormalizer:
    """Handles text normalization operations."""
    
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words: Set[str] = set(stopwords.words('english'))
        
    def normalize(self, text: str) -> str:
        """Normalize text by removing articles, punctuation, and extra whitespace."""
        text = self._remove_articles(text)
        text = self._remove_punctuation(text)
        text = self._fix_whitespace(text)
        return text.lower()
    
    def get_tokens(self, text: str) -> List[str]:
        """Convert text to normalized tokens."""
        return self.normalize(text).split() if text else []
    
    def get_tokens_without_stopwords(self, text: str) -> List[str]:
        """Get normalized tokens with stopwords removed."""
        tokens = self.get_tokens(text)
        return [token for token in tokens if token not in self.stop_words]
    
    @staticmethod
    def _remove_articles(text: str) -> str:
        """Remove articles (a, an, the) from text."""
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    @staticmethod
    def _fix_whitespace(text: str) -> str:
        """Standardize whitespace in text."""
        return ' '.join(text.split())

@dataclass
class ScoringResult:
    """Data class to hold various scoring metrics."""
    precision: float
    recall: float
    f1: float

class TextScorer:
    """Handles various text scoring metrics."""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def compute_exact_match(self, reference: str, prediction: str) -> int:
        """Compute exact match score."""
        return int(reference.lower() in prediction.lower())
    def comput_em(self, reference:str, prediction:str):
        ref_normalized = self.normalizer.normalize(reference)
        pred_normalized = self.normalizer.normalize(prediction)
        return ref_normalized in pred_normalized
    def compute_rouge_l(self, reference: str, prediction: str) -> float:
        """Compute ROUGE-L F1 score."""
        ref_normalized = self.normalizer.normalize(reference)
        pred_normalized = self.normalizer.normalize(prediction)
        scores = self.rouge_scorer.score(ref_normalized, pred_normalized)
        return scores['rougeL'][2]  # Return F1 score
    
    def compute_f1(self, a_gold, a_pred):
        """Compute F1 score between two strings.

        Args:
          a_gold: string one
          a_pred: string two

        Returns:
              f1 score
        """

        gold_toks = self.normalizer.get_tokens(a_gold)
        pred_toks = self.normalizer.get_tokens(a_pred)

        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if len(gold_toks) == 0 or len(pred_toks) == 0:
          # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
          return int(gold_toks == pred_toks)

        if num_same == 0:
          return 0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
    
    def _get_removed_tokens(self, text: str, from_texts: List[str]) -> List[str]:
        """Get tokens from text with tokens from from_texts removed."""
        gold_tokens = self.normalizer.get_tokens_without_stopwords(text)
        from_tokens = set()
        for from_text in from_texts:
            from_tokens.update(self.normalizer.get_tokens_without_stopwords(from_text))
        
        return [token for token in gold_tokens if token not in from_tokens]

class Evaluator:
    """Main evaluation class that coordinates scoring across multiple metrics."""
    
    def __init__(self):
        self.scorer = TextScorer()
        
    def load_data(self, qa_path: str, asqa_path: str, processed_qa_path:str) -> Tuple[Dict, Dict, Dict]:
        """Load and process the required data files."""
        with open(qa_path, 'r') as f:
            train = json.load(f)
        with open(asqa_path, 'r') as f:
            annotation = json.load(f)
        with open(processed_qa_path,'r') as f:
            processed = json.load(f)
            
        id2ans = {}
        id2gen = {}
        id2context = {}
        cnt = 0
        id2doc = {}
        
        for ex in train["data"]:
            id2ans[ex["id"]] = ex["answers"]["text"]
            preid, _ = ex["id"].split("_")
            for idx, anno in enumerate( annotation):
                if preid == anno["sample_id"]:
                    id2gen[preid] = [ann['long_answer'] for ann in anno['annotations']]
                    id2doc[preid] =  processed[idx]["documents"]
                    break
            id2context[preid] = ex["context"]
        return id2ans, id2gen, id2context, id2doc
    
    def evaluate_predictions(self, predictions: Dict, id2ans: Dict, 
                           id2gen: Dict, id2context: Dict, id2doc:Dict) -> Dict:
        """Evaluate predictions using multiple metrics."""
        scores_f1 = {}
        scores_rouge = {}
        scores_coverage = {}
        score_em = {}
        # Calculate F1 and coverage scores
        for idx, gen_ans in predictions.items():
            preid, _ = idx.split("_")
            
            if preid not in scores_f1:
                scores_f1[preid] = []
                scores_coverage[preid] = []
                score_em[preid] = []
                
            max_f1 = -1
            current_cov = 0
            current_em = 0
            long_ans = id2context[preid]
            for ref_ans in id2ans[idx]:
                current_f1 = self.scorer.compute_f1(ref_ans, gen_ans)
                max_f1 = max(max_f1, current_f1)
                for doc in id2doc[preid]:
                    if ref_ans in doc:
                        current_cov = 1
                if self.scorer.comput_em(ref_ans, long_ans):
                    current_em = 1
            score_em[preid].append(current_em)   
            scores_coverage[preid].append(current_cov) 
            scores_f1[preid].append(max_f1)
            
        
        # Calculate ROUGE scores
        for idx, gen_ans in id2context.items():
            scores_rouge[idx] = []
            max_rouge = -1
            for ans in id2gen[idx]:
                curr = self.scorer.compute_rouge_l(ans, gen_ans)
                max_rouge = max(max_rouge, curr)
            scores_rouge[idx].append(max_rouge)
        
        # Calculate averages
        avg_f1 = {preid: sum(scores)/len(scores) for preid, scores in scores_f1.items()}
        avg_rouge = {preid: sum(scores)/len(scores) for preid, scores in scores_rouge.items()}
        avg_coverage = {preid: sum(scores)/len(scores) for preid, scores in scores_coverage.items()}
        avg_em = {preid: sum(scores)/len(scores) for preid, scores in score_em.items()}
        overall_f1 = sum(avg_f1.values()) / len(avg_f1)
        overall_rouge = sum(avg_rouge.values()) / len(avg_rouge)
        overall_coverage = sum(avg_coverage.values()) / len(avg_coverage)
        overall_em = sum(avg_em.values())/len(avg_em)
        return {
            "f1_scores": scores_f1,
            "rouge_scores": scores_rouge,
            "coverage_scores": scores_coverage,
            "average_f1": avg_f1,
            "average_rouge": avg_rouge,
            "average_coverage": avg_coverage,
            "overall_f1": overall_f1,
            "overall_rouge": overall_rouge,
            "overall_coverage": overall_coverage,
            "overall_em":overall_em
        }

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Text Evaluation System')
    
    # Required arguments
    parser.add_argument('--processed-path', required=True)
    parser.add_argument('--qa-path', required=True,
                      help='Path to the QA dataset JSON file')
    parser.add_argument('--asqa-path', required=True,
                      help='Path to the ASQA dataset JSON file')
    parser.add_argument('--predictions-path', required=True,
                      help='Path to the predictions JSON file')
    
    # Optional arguments
    parser.add_argument('--output-path', 
                      help='Path to save detailed results JSON file (optional)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed scores for each example')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    try:
        # Load data
        print("Loading datasets...")
        id2ans, id2gen, id2context, id2doc = evaluator.load_data(
            qa_path=args.qa_path,
            asqa_path=args.asqa_path,
            processed_qa_path=args.processed_path
        )
        
        # Load predictions
        print("Loading predictions...")
        with open(args.predictions_path, "r") as f:
            predictions = json.load(f)
        
        # Run evaluation
        print("Running evaluation...")
        results = evaluator.evaluate_predictions(predictions, id2ans, id2gen, id2context,id2doc)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Overall F1 Score: {results['overall_f1']:.4f}")
        print(f"Overall ROUGE-L Score: {results['overall_rouge']:.4f}")
        print(f"Overall Coverage Score: {results['overall_coverage']:.4f}")
        print(f"Overall EM Score: {results['overall_em']:.4f}")
        # Print detailed scores if verbose mode is enabled
        if args.verbose:
            print("\nDetailed Scores:")
            print("\nAverage F1 scores per example:")
            for preid, score in results['average_f1'].items():
                print(f"ID {preid}: {score:.4f}")
                
            print("\nAverage ROUGE-L scores per example:")
            for preid, score in results['average_rouge'].items():
                print(f"ID {preid}: {score:.4f}")
        
        # Save detailed results if output path is provided
        if args.output_path:
            print(f"\nSaving detailed results to {args.output_path}")
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()