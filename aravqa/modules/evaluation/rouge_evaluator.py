from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class ROUGEEvaluator(BaseEvaluator):
    def __init__(self):
        self.rouge_scorer = evaluate.load("rouge")
        super().__init__()

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using ROUGE.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_rouge": {"rouge1":float('-inf'),"rouge2":float('-inf'),"rougeL":float('-inf')},
            "rouge": []
        }

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating ROUGE scores"):
            rouge_scores = self._compute_rouge_score([pred], [ref])
            results["rouge"].append(rouge_scores)
           
        overall_scores= self._compute_rouge_score(predictions,references)

        results["overall_rouge"]["rouge1"] = overall_scores["rouge1"]
        results["overall_rouge"]["rouge2"] = overall_scores["rouge2"]
        results["overall_rouge"]["rougeL"] = overall_scores["rougeL"]

        return results

    def _compute_rouge_score(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using ROUGE.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score. {"rouge1":float("-inf"),"rouge2":float('-inf'),"rougeL":float('-inf')} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

        try:
            result = self.rouge_scorer.compute(predictions=predictions, references=references)
            return result
        except Exception as e:
            print(f"Error computing ROUGE score: {e}")
            return {"rouge1":float("-inf"),"rouge2":float('-inf'),"rougeL":float('-inf')}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)