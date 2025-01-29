from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
import evaluate
import numpy as np

class SQuADv2Evaluator(BaseEvaluator):
    def __init__(self, no_answer_threshold: float = 0.5):
        self.squad_v2_scorer = evaluate.load("squad_v2")
        super().__init__()
        self.no_answer_threshold = no_answer_threshold

    def evaluate(self, predictions: List[str], references: List[List[str]], has_answer_predictions: List[bool]) -> Dict:
        """
        Evaluates predictions against references using SQuAD v2 metrics (EM, F1, No Answer).

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[List[str]]): List of lists of reference answers.
            has_answer_predictions (List[bool]): List of boolean if the prediction predicted an answer.
        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "overall_em": float('-inf'),
            "overall_f1": float('-inf'),
            "em": [],
            "f1": [],
             "overall_no_answer_probability_threshold":float("-inf"),
        }

        for pred, ref_list,has_answer in tqdm(zip(predictions, references,has_answer_predictions), total=len(predictions), desc="Evaluating SQuAD v2 scores"):
            ref = ref_list[0] #SQuAD metric takes a single reference

            squad_score = self._compute_squad_v2_score([pred], [ref], [has_answer])
            results["em"].append(squad_score["exact_match"])
            results["f1"].append(squad_score["f1"])

        overall = self._compute_squad_v2_score(predictions, [ref[0] for ref in references], has_answer_predictions)
        results["overall_em"] = overall["exact_match"]
        results["overall_f1"] = overall["f1"]
        results["overall_no_answer_probability_threshold"] = overall["no_answer_probability_threshold"]


        return results

    def _compute_squad_v2_score(self, predictions: List[str], references: List[str], has_answer_predictions:List[bool]) -> Dict:
        """
        Evaluates predictions against a list of references using SQuAD v2 metrics.

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[str]): List of reference answers.
            has_answer_predictions (List[bool]): List of boolean if the prediction predicted an answer.

        Returns:
            Dict: The computed evaluation score. {"exact_match": float("-inf"), "f1": float("-inf"), "no_answer_probability_threshold":float("-inf")} if computation fails.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references) or len(predictions) != len(has_answer_predictions):
             raise ValueError("The number of predictions, reference and has_answer_predictions must match.")

        try:
            result = self.squad_v2_scorer.compute(predictions=predictions, references=references, no_answer_probs=np.array(has_answer_predictions, dtype=float))
            return result
        except Exception as e:
            print(f"Error computing SQuAD v2 score: {e}")
            return {"exact_match": float("-inf"), "f1": float("-inf"), "no_answer_probability_threshold":float("-inf")}

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)