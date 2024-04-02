import asyncio
import logging
import random
import re
from datetime import timedelta
from io import BytesIO
from typing import List, Optional, Dict, Any, Literal

from PIL import Image, UnidentifiedImageError
from replicate import Client
from replicate.prediction import Prediction
from result import Result, Err, Ok
from temporalio import activity, workflow

from ImageUtil import ImageUtil

logger = logging.getLogger(__name__)


class Replicate:
    def __init__(self, api_key):
        self.client = Client(
            api_token=api_key,
        )

    def get_activities(self) -> List:
        return [
            self._get_prediction,
            self._create_prediction,
        ]

    @staticmethod
    def open_file(image_bytes: bytes) -> Result[Image, str]:
        try:
            image = Image.open(BytesIO(image_bytes))
        except UnidentifiedImageError:
            return Err('Данный формат изображения не поддерживается')

        return Ok(image)

    @staticmethod
    def to_media_string_path(base64_file: str) -> str:
        return f"data:application/octet-stream;base64,{base64_file}"

    @activity.defn
    async def _create_prediction(
        self,
        ref: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> dict:
        try:
            match = re.match(r"^(?P<owner>[^/]+)/(?P<name>[^:]+):(?P<version>.+)$", ref)

            if not match:
                raise Exception(
                    f"Invalid reference to model version: {ref}. Expected format: owner/name:version"
                )

            version_id = match.group("version")

            response = await self.client.predictions.async_create(
                version=version_id,
                input=data or {}
            )
        except Exception as e:
            logger.error(f"Error running model: {e}")

            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response,
            'err': None,
        }

    @staticmethod
    async def creative_upscaler(
        image: str,
        prompt: str = "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
        dynamic: float = 6,
        sd_model: str = "juggernaut_reborn.safetensors [338b85bc4f]",
        scheduler: str = "DPM++ 3M SDE Karras",
        creativity: float = 0.35,
        lora_links: str = "",
        downscaling: bool = False,
        resemblance: float = 0.6,
        scale_factor: float = 2,
        tiling_width: int = 112,
        tiling_height: int = 144,
        custom_sd_model: str = "",
        negative_prompt: str = "(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
        num_inference_steps: int = 18,
        downscaling_resolution: int = 768,
        seed: int = None
    ) -> Result[str, str]:
        """
           Upscales an image creatively based on the provided parameters.

           Args:
             image (str): Path to the input image.
             prompt (str, optional): Prompt for guiding the upscaling. Defaults to "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>".
             dynamic (float, optional): HDR parameter. Defaults to 6.
             sd_model (str, optional): Stable Diffusion model checkpoint. Defaults to "juggernaut_reborn.safetensors [338b85bc4f]".
             scheduler (str, optional): Scheduler for the diffusion process. Defaults to "DPM++ 3M SDE Karras".
             creativity (float, optional): Creativity parameter. Defaults to 0.35.
             lora_links (str, optional): Links to Lora files for additional details. Defaults to "".
             downscaling (bool, optional): Whether to downscale the image before upscaling. Defaults to False.
             resemblance (float, optional): Resemblance parameter. Defaults to 0.6.
             scale_factor (float, optional): Scale factor for upscaling. Defaults to 2.
             tiling_width (int, optional): Width of tiles for fractality. Defaults to 112.
             tiling_height (int, optional): Height of tiles for fractality. Defaults to 144.
             custom_sd_model (str, optional): Link to a custom Stable Diffusion model. Defaults to "".
             negative_prompt (str, optional): Negative prompt for guidance. Defaults to "(worst quality, low quality, normal quality:2) JuggernautNegative-neg".
             num_inference_steps (int, optional): Number of denoising steps. Defaults to 18.
             downscaling_resolution (int, optional): Resolution for downscaling. Defaults to 768.
             seed (int, optional): Random seed. Defaults to 1337.
           """

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        result = await workflow.execute_activity(
            Replicate._create_prediction,
            args=[
                "philz1337x/clarity-upscaler:1d336fed07b3421d5dc9e3bf8ef96a88429f61e34665ceb5cbb0d747425d8368",
                {
                    "image": image,
                    "prompt": prompt,
                    "dynamic": dynamic,
                    "sd_model": sd_model,
                    "scheduler": scheduler,
                    "creativity": creativity,
                    "lora_links": lora_links,
                    "downscaling": downscaling,
                    "resemblance": resemblance,
                    "scale_factor": scale_factor,
                    "tiling_width": tiling_width,
                    "tiling_height": tiling_height,
                    "custom_sd_model": custom_sd_model,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "downscaling_resolution": downscaling_resolution,
                    "seed": seed
                }
            ],
            start_to_close_timeout=timedelta(seconds=600)
        )

        if result['err']:
            return Err(result['err'])

        prediction = result['ok']

        result = await Replicate.wait_for_prediction(prediction['id'])

        return result

    @staticmethod
    async def face_to_many(
        image: str,
        style: Literal['3D', 'Emoji', 'Video game', 'Pixels', 'Clay', 'Toy'],
        prompt: str = "a person",
        lora_scale: float = 1,
        negative_prompt: str = "",
        prompt_strength: float = 4.5,
        denoising_strength: float = 0.65,
        instant_id_strength: float = 0.8,
        control_depth_strength: float = 0.8,
    ) -> Result[str, str]:
        result = await workflow.execute_activity(
            Replicate._create_prediction,
            args=[
                "fofr/face-to-many:35cea9c3164d9fb7fbd48b51503eabdb39c9d04fdaef9a68f368bed8087ec5f9",
                {
                    "image": image,
                    "style": style,
                    "prompt": prompt,
                    "lora_scale": lora_scale,
                    "negative_prompt": negative_prompt,
                    "prompt_strength": prompt_strength,
                    "denoising_strength": denoising_strength,
                    "instant_id_strength": instant_id_strength,
                    "control_depth_strength": control_depth_strength,
                }
            ],
            start_to_close_timeout=timedelta(seconds=600)
        )

        if result['err']:
            return Err(result['err'])

        prediction = result['ok']

        result = await Replicate.wait_for_prediction(prediction['id'])

        return result

    @activity.defn
    async def _get_prediction(self, prediction_id: str) -> dict:
        return {
            'ok': await self.client.predictions.async_get(prediction_id),
            'err': None,
        }

    @staticmethod
    async def get_prediction(prediction_id: str) -> Result[Prediction, str]:
        result = await workflow.execute_activity(
            Replicate._get_prediction,
            args=[prediction_id],
            start_to_close_timeout=timedelta(seconds=30)
        )

        if result['err'] is not None:
            return Err(result['err'])

        prediction = Prediction(**result['ok'])
        return Ok(prediction)

    @staticmethod
    async def wait_for_prediction(prediction_id: str, wait_for=600) -> Result[dict, str]:
        i = 0

        while i < wait_for:
            logger.info(f"Waiting for prediction {prediction_id} to complete")

            prediction = await Replicate.get_prediction(prediction_id)

            if prediction.is_err():
                return Err(prediction.err())

            prediction = prediction.ok()

            logger.info(f"Prediction {prediction_id} progress: {prediction.progress}")

            if prediction.status in ["failed", "canceled"]:
                logger.error(f"Prediction {prediction_id} failed: {prediction.error}")

                return Err(f"Prediction {prediction_id} failed: {prediction.error}")

            if prediction.status in ["succeeded"]:
                logger.info(f"Prediction {prediction_id} completed")

                return Ok(prediction.output)

            await asyncio.sleep(10)
            i += 1

        return Err(f"Prediction {prediction_id} failed to complete in {wait_for} seconds")