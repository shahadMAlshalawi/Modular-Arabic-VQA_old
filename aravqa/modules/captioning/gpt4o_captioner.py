import torch
from typing import List, Union, Iterable, Dict
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import base64
from openai import OpenAI
from google.colab import userdata


class GPT4oCaptioner:
    """
    Implementation of image captioning using GPT-4o via the OpenAI API,
    without inheriting from BaseCaptioner.
    """

    def __init__(self, api_key: str, model_name: str=GPT4oConfig.MODEL_NAME, max_tokens: int = GPT4oConfig.MAX_LENGTH):
        """
        Initializes the GPT-4o captioning model.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the GPT-4o model to use (default: "gpt-4o").
            max_tokens (int): The maximum number of tokens to generate in the caption.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
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
        Generates captions for a batch of images using the GPT-4o API.

        Args:
            images: A list of input images.

        Returns:
            List[List[Dict[str, str]]]: A list of lists (one list per image) containing dictionaries with the generated captions.
        """
        prepared_images = self.prepare_images(images)
        captions = []
        for image in prepared_images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = "data:image/jpeg;base64," + str(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that generates accurate image captions in Arabic."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": img_str, "detail": "high"}},
                        {"type": "text", "text": "Describe this image in six short and detailed captions in Arabic, each on a new line and do not use bullet point symbols"},
                    ]}
                ],
                max_tokens=self.max_tokens,
            )
            caption_text = response.choices[0].message.content
            caption_lines = [line for line in caption_text.split('\n') if line.strip()]
            captions.append([{"caption": line.strip(), "id": idx} for idx, line in enumerate(caption_lines)])
        return captions

    def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
        """
        Generates captions for a batch of images using the GPT-4o API.

        Args:
            images: A list of input images.

        Returns:
            List[List[Dict[str, str]]]: A list of lists (one list per image) containing dictionaries with the generated captions.
        """
        return self.generate_caption(images)