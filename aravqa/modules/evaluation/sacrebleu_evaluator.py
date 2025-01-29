from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class SacreBLEUEvaluator(BaseEvaluator):
    def __init__(self):
        self.sacrebleu_scorer = evaluate.load("sacrebleu")
        super().__init__()

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using SacreBLEU.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_sacrebleu": float('-inf'),
            "sacrebleu": []
        }

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating SacreBLEU scores"):
            sacrebleu = self._compute_sacrebleu_score([pred], [ref])["score"]
            results["sacrebleu"].append(sacrebleu)
           
        overall= self._compute_sacrebleu_score(predictions,references)
        results["overall_sacrebleu"] = overall["score"]

        return results

    def _compute_sacrebleu_score(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using SacreBLEU.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score. {"score":float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

        try:
            result = self.sacrebleu_scorer.compute(predictions=predictions, references=references)
            return result
        except Exception as e:
            print(f"Error computing SacreBLEU score: {e}")
            return {"score":float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)