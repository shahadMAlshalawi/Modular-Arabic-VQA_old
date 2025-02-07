from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class JaccardEvaluator(BaseEvaluator):
    def __init__(self, max_len: int = 1):
        self.jaccard_scorer = evaluate.load("jaccard_similarity")
        super().__init__()
        self.max_len = max_len

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using Jaccard.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_jaccard": float('-inf'),
            "jaccard": []
        }

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating Jaccard scores"):
            jaccard = self._compute_jaccard_score([pred], [ref])["jaccard"]
            results["jaccard"].append(round(jaccard,3))
        
        overall= self._compute_jaccard_score(predictions,references)
        results["overall_jaccard"] = round(overall["jaccard"],3)


        return results

    def _compute_jaccard_score(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using Jaccard.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score. {"jaccard": float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

        try:
            result = self.jaccard_scorer.compute(predictions=predictions, references=references, max_len=self.max_len)
            # TODO: Spacify tokenizer like result = self.jaccard_scorer(predictions=predictions, references=references,tokenizer=..._tokenize, max_order=self.max_order)
            return result
        except Exception as e:
            print(f"Error computing Jaccard score: {e}")
            return {"jaccard": float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)