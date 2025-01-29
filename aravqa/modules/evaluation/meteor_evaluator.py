from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class METEOREvaluator(BaseEvaluator):
    def __init__(self, language: str = "ar"):
        self.meteor_scorer = evaluate.load("meteor")
        super().__init__()
        self.language=language

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using METEOR.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_meteor": float('-inf'),
            "meteor": []
        }

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating METEOR scores"):
            meteor = self._compute_meteor_score([pred], [ref])["meteor"]
            results["meteor"].append(meteor)
        
        overall = self._compute_meteor_score(predictions, references)
        results["overall_meteor"] = round(overall["meteor"],2)

        return results

    def _compute_meteor_score(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using METEOR.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score. {"meteor": float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
             raise ValueError("The number of predictions must match the number of reference sets.")
        try:
            result = self.meteor_scorer.compute(predictions=predictions, references=references)
            return result
        except Exception as e:
            print(f"Error computing METEOR score: {e}")
            return {"meteor": float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)