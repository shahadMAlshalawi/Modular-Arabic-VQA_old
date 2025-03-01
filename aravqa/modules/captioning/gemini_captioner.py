import torch
from typing import List, Union, Iterable, Dict
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import base64

import google.generativeai as genai

class GeminiCaptioner:
    """
    Implementation of image captioning using Gemini 2.0 Flash via the Google AI Gemini API.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", max_tokens: int = 1024):
        """
        Initializes the Gemini 2.0 Flash captioning model.

        Args:
            api_key (str): Google AI Gemini API key.
            model_name (str): The name of the Gemini model to use (default: "gemini-1.5-flash").
            max_tokens (int): The maximum number of tokens to generate in the caption.
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_image(self, image: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
        """
        Prepares a single image for caption generation, converting it to RGB format. Handles various input types.

        Args:
            image: The input image. Can be a file path, URL, NumPy array, PyTorch tensor, or PIL Image.

        Returns:
            A PIL Image object in RGB format.

        Raises:
            ValueError: If the input image type is unsupported or if an error occurs during image processing.
            requests.exceptions.RequestException: If there's an error downloading the image from a URL.
        """

        try:

            if isinstance(image, Image.Image):
                return image.convert("RGB")

            elif isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    return Image.open(image).convert("RGB")

            elif isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")

            elif torch.is_tensor(image):
                return Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

            else:
                raise ValueError(f"Unsupported image input type: {type(image)}")

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error downloading image: {e}")

    def prepare_images(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[Image.Image]:
        """
        Prepares a batch of images for caption generation by converting them to PIL Images in RGB format.

        Args:
            images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.

        Returns:
            A list of PIL Image objects in RGB format.
        """

        if not isinstance(images, list):
            images = [images]

        return [self._prepare_image(image) for image in images]

    def generate_caption(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
        """
        Generates captions for a batch of images using the Gemini API.

        Args:
            images: A list of input images.

        Returns:
            List[List[Dict[str, str]]]: A list of lists (one list per image) containing dictionaries with the generated captions.
        """
        prepared_images = self.prepare_images(images)
        captions = []
        for image in prepared_images:
            prompt_parts = [
                """
                You are an AI assistant that generates accurate image captions in Arabic. Describe this image in three short and detailed captions in Arabic, each on a new line and do not use bullet point symbols.
                """,
                image
            ]

            try:
                response = self.model.generate_content(prompt_parts, safety_settings={
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                }, generation_config={
                    "max_output_tokens": self.max_tokens
                })
                caption_text = response.text

                caption_lines = [line for line in caption_text.split('\n') if line.strip()]
                captions.append([{"caption": line.strip(), "id": idx} for idx, line in enumerate(caption_lines)])

            except Exception as e:
                print(f"Error generating caption: {e}")
                captions.append([{"caption": "Error generating caption", "id": 0}]) #Add error caption if failed


        return captions

    def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
        """
        Generates captions for a batch of images using the Gemini API.

        Args:
            images: A list of input images.

        Returns:
            List[List[Dict[str, str]]]: A list of lists (one list per image) containing dictionaries with the generated captions.
        """
        return self.generate_caption(images)