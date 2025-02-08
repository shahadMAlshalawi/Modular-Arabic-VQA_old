#This module is used to evaluate the semantic similarity between the predicted answers and the ground truth answers. 
#The module uses the OpenAI API for embedding creation (model="text-embedding-ada-002") for ground truth and the predicted answers. 
#The Cosine simislarity between the ebeddings is calculated using sklearn's cosine_similarity function.
from typing import List, Dict
from tqdm import tqdm
from .base import BaseEvaluator
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class SemanticSimilarityEvaluator(BaseEvaluator):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client
        super().__init__()

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references using semantic similarity.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """

        results = {
            "overall_similarity": float('-inf'),
            "similarities": []
        }

        for pred, ref_list in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating Semantic Similarities"):
            similarity = self._compute_similarity_score(pred, ref_list)
            results["similarities"].append(round(similarity, 3))

        overall_similarities = [self._compute_similarity_score(pred, ref_list) for pred, ref_list in zip(predictions, references)]
        results["overall_similarity"] = round(np.mean(overall_similarities), 3) # Use mean for overall score


        return results

    def _compute_similarity_score(self, prediction: str, references: List[str]) -> float:
        """
        Computes the average of the top 3 semantic similarity distances between a prediction and a list of references.

        Args:
            prediction (str): The predicted sentence.
            references (List[str]): A list of reference sentences.

        Returns:
            float: The average of the top 3 semantic similarity distances. Returns -inf if computation fails or if there are less than 3 references.
        """
        if not prediction or not references:
            print("Warning: Empty prediction or references encountered.")
            return float('-inf')

        try:
            distances = []
            for ref in references:
                similarity_info = self._evaluate_similarity(prediction, ref)
                similarity = similarity_info["similarity"]
                distance = 1 - similarity  # Convert similarity to distance
                distances.append(distance)

            if len(distances) < 3:
                print("Warning: Less than 3 references available.")
                return float('-inf')

            top_3_distances = sorted(distances)[:3]  # Get the smallest 3 distances (highest similarity)
            avg_distance = np.mean(top_3_distances)
            return 1 - avg_distance # convert back to similarity


        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return float('-inf')

    def _evaluate_similarity(self, text1, text2):
        # Get embeddings using client.embeddings.create and return full response
        response1 = self.client.embeddings.create(input=text1, model="text-embedding-ada-002")
        response2 = self.client.embeddings.create(input=text2, model="text-embedding-ada-002")

        # Extract embeddings from responses
        embedding1 = np.array(response1.data[0].embedding).reshape(1, -1)
        embedding2 = np.array(response2.data[0].embedding).reshape(1, -1)

        similarity = self.cosine_similarity(embedding1, embedding2)[0][0]

        # Return full responses along with similarity score
        return {
            "response1": response1,
            "response2": response2,
            "similarity": similarity
        }

    def cosine_similarity(self, vec1, vec2):
        """
        Calculates the cosine similarity using scikit-learn's cosine_similarity function.
        """
        return cosine_similarity(vec1, vec2)

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)