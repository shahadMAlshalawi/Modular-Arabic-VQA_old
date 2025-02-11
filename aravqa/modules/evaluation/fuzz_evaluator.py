import os
import json
from .base import BaseEvaluator
from typing import List, Dict
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI

class AnswerScore(BaseModel):
    score: int

class FuzzEvaluator(BaseEvaluator):
    def __init__(self, openai_api_key: str):
        """
        Initializes the FuzzEvaluator.

        Args:
            openai_api_key (str): The OpenAI API key.
        """
        self.openai_api_key = openai_api_key
        super().__init__()

    def evaluate(self, predictions: List[str], references: List[List[str]], questions: List[str]) -> Dict:
        """
        Evaluates predictions against references using GPT-4o for semantic similarity.

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[List[str]]): List of lists of reference answers.
            questions (List[str]): List of questions corresponding to each prediction and reference.

        Returns:
            Dict: The computed evaluation score.
        """
        results = {
            "fuzz_overall_accuracy": float('-inf'),
            "fuzz_accuracy": []
        }

        evaluated_data = []
        for i, (pred, ref_list, question) in tqdm(enumerate(zip(predictions, references, questions)), total=len(predictions), desc="Evaluating with GPT-4o"):
          evaluation = self._compute_fuzz_score(question=question, pred=pred, gt_list=ref_list,index=i)
          evaluated_data.append(evaluation)

        correct_count = sum(1 for item in evaluated_data if item['evaluation'] == 1)
        total_count = len(evaluated_data)
        accuracy = correct_count / total_count

        results["fuzz_overall_accuracy"] = round(accuracy, 3)

        results["fuzz_accuracy"] = evaluated_data["evaluation"]


        return results

    def _compute_fuzz_score(self, question: str, pred: str, gt_list: List[str], index: int) -> Dict:
        """
        Computes the fuzz score between a prediction and a list of references using GPT-4o.

        Args:
            question (str): The question corresponding to the prediction and reference.
            pred (str): The predicted answer.
            gt_list (List[str]): The list of ground truth answers.
            index (int) The index of the sample.

        Returns:
            Dict: The evaluation score. {"index": int, "question": str, "pred_answer": str, "answers": str, "evaluation": int}
        """

        try:
            client = OpenAI(api_key=self.openai_api_key)
            gt_str = ",".join(gt_list)

            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in natural language understanding and semantic similarity. Your task is to evaluate the semantic similarity between two given sentences: a predicted answer and ground truth answers. You should output a score of 1 if the sentences are semantically similar, and 0 if they are not.""",
                },
                {
                    "role": "user",
                    "content": f"""Here are three examples to guide your evaluation:
        Example 1:
        Question: "ما هي اللغة المستخدمة في النص؟"
        Predicted Answer: "العربية"
        Ground Truth: "عربيه","عربي","لغة عربية","العربيه","اللغة العربية","عربية","عربي","لغه عربيه","العربية","اللغة العربية"
        Score: 1

        Example 2:
        Question: "ما هو موضوع النص؟"
        Predicted Answer: "إثنان"
        Ground Truth: "الحب و الكراهية","حب","كراهية","الحب","الكراهية","حب و كراهية""الكراهية والحب","كراهية وحب","حب","كراهيه"
        Score: 0

        Example 3:
        Question: "ما هو عدد صفحات الكتاب؟"
        Predicted Answer: "الصورة لا تحتوي على عدد صفحات الكتاب."
        Ground Truth: "غير واضح","لا يتضح","لا يمكن معرفة هذا","لا يمكن تحديد ذلك","لا يمكن تحديد هذا","لا يعرف هذا من الصورة","لا يمكن معرفة هذا من الصورة","لا يتضح من الصورة","لا يمكن تحديده","غير معروف"
        Score: 1

        Now, for each new pair of sentences, analyze their semantic similarity and provide a score of 1 for similar meanings or 0 for different meanings. Always consider the context and potential variations in expressing the same concept.
        Question: "{question}"
        Predicted Answer: "{pred}"
        Ground Truth: "{gt_str}"
        Score: """
                },
            ]

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "answer_score",
                            "description": "Provide a [0, 1] score to the semantic similarity between two sentences",
                            "parameters": AnswerScore.model_json_schema(),
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "answer_score"}},
            )

            vqa_answer = AnswerScore.model_validate_json(
                completion.choices[0].message.tool_calls[0].function.arguments
            )
            return {
                'index': index,
                'question': question,
                'pred_answer': pred,
                'answers': gt_str,
                'evaluation': vqa_answer.score
            }
        except Exception as e:
            print(f"Error computing Fuzz score: {e}")
            return {
                'index': index,
                'question': question,
                'pred_answer': pred,
                'answers': gt_str,
                'evaluation': float('-inf')
            }

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)