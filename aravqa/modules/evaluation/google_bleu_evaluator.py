from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate

class GoogleBLEUEvaluator(BaseEvaluator):
    def __init__(self, max_len: int = 1):
        self.google_bleu_scorer = evaluate.load("google_bleu")
        super().__init__()
        self.max_len = max_len

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using Google BLEU.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_google_bleu": float('-inf'),
            "google_bleu": []
        }

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating Google BLEU scores"):
            google_bleu = self._compute_google_bleu_score([pred], [ref])["google_bleu"]
            results["google_bleu"].append(round(google_bleu,3))
        
        overall= self._compute_google_bleu_score(predictions,references)
        results["overall_google_bleu"] = round(overall["google_bleu"],3)


        return results

    def _compute_google_bleu_score(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using Google BLEU.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score. {"google_bleu": float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

        try:
            result = self.google_bleu_scorer.compute(predictions=predictions, references=references, max_len=self.max_len)
            # TODO: Spacify tokenizer like result = self.google_bleu_scorer(predictions=predictions, references=references,tokenizer=..._tokenize, max_order=self.max_order)
            return result
        except Exception as e:
            print(f"Error computing Google BLEU score: {e}")
            return {"google_bleu": float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)