from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate
import numpy as np

class BERTScoreEvaluator(BaseEvaluator):
    def __init__(self, lang: str = "ar", model_type: str = "distilbert-base-multilingual-cased"):
        # TODO: Spacify layer: if model_type: str = "bert-base-multilingual-cased" with num_layers=9
        # TODO: Another model option: model_type: str = "distilbert-base-multilingual-cased" with num_layers=5
        self.bertscore_scorer = evaluate.load("bertscore")
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
          results["precision_bertscore"].append(round(bertscore["precision"][0], 3))
          results["recall_bertscore"].append(round(bertscore["recall"][0],3))
          results["f1_bertscore"].append(round(bertscore["f1"][0], 3))

        overall= self._compute_bertscore(predictions,references)
        results["overall_precision_bertscore"]=round(np.mean(overall["precision"]),3)
        results["overall_recall_bertscore"]=round(np.mean(overall["recall"]),3)
        results["overall_f1_bertscore"]= round(np.mean(overall["f1"]),3)
        
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
            result = self.bertscore_scorer.compute(predictions=predictions, references=references, lang=self.lang, model_type=self.model_type)
            # TODO: Spacify layer: if model_type: str = "bert-base-multilingual-cased" with num_layers=9 if model_type: str = "distilbert-base-multilingual-cased" with num_layers=5
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