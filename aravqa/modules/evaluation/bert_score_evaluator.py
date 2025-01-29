from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class BERTScoreEvaluator(BaseEvaluator):
    def __init__(self, lang: str = "ar", model_type: str = "bert-base-multilingual-cased"):
        # Another option: model_type: str = "distilbert-base-multilingual-cased"
        self.bertscore_scorer = evaluate.load("bertscore", lang=lang, model_type=model_type)
        super().__init__()
        self.lang = lang
        self.model_type = model_type
    
    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using BERTScore.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_precision_bertscore":float('-inf'),
            "overall_recall_bertscore":float('-inf'),
            "overall_f1_bertscore":float('-inf'),
            "precision_bertscore": [],
            "recall_bertscore": [],
            "f1_bertscore": []
        }
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating BERTScore scores"):
          bertscore = self._compute_bertscore([pred], [ref])
          results["precision_bertscore"].append(bertscore["precision"])
          results["recall_bertscore"].append(bertscore["recall"])
          results["f1_bertscore"].append(bertscore["f1"])

        overall= self._compute_bertscore(predictions,references)
        results["overall_precision_bertscore"]=overall["precision"]
        results["overall_recall_bertscore"]=overall["recall"]
        results["overall_f1_bertscore"]= overall["f1"]
        
        return results

    def _compute_bertscore(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using BERTScore.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.  {"precision":float("-inf"),"recall":float('-inf'),"f1":float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

        try:
            result = self.bertscore_scorer.compute(predictions=predictions, references=references)
            return result
        except Exception as e:
            print(f"Error computing BERTScore score: {e}")
            return {"precision":float("-inf"),"recall":float('-inf'),"f1":float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)