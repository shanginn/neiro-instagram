import asyncio
import base64
import logging
from datetime import timedelta
from io import BytesIO
from typing import List, Optional

import aiohttp
from PIL import Image, UnidentifiedImageError, ImageFile
from result import Result, Ok, Err
from novita_client import UpscaleRequest, UpscaleResponse, ProgressResponse, ReimagineRequest, ReimagineResponse, \
    NovitaResponseError, MergeFaceResponse, RemoveBackgroundRequest, RemoveBackgroundResponse, MakePhotoLoRA, \
    MakePhotoRequest, MakePhotoResponse, V3TaskResponse, Txt2ImgRequest, Txt2ImgResponse
from novita_client import NovitaClient as BaseNovitaClient
from temporalio import activity, workflow

from ImageUtil import ImageUtil

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class NovitaClient:
    def __init__(self, api_key):
        self.client = BaseNovitaClient(api_key)

    def get_activities(self) -> List:
        return [
            self._upscale,
            self._create_dreambooth_task,
            self._novita_progress,
            self._remove_background,
            self._swap_faces,
            self._place_image_on_size_activity,
            self._replace_background,
            self._upload_assets,
            self._task_result_v3,
            self._create_text2img_task,
            self._task_progress_v2,
        ]

    @activity.defn
    async def _upscale(self, request_dict: dict, file_url: Optional[str] = None) -> dict:
        request = UpscaleRequest.from_dict(request_dict)
        if file_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url) as response:
                    file_bytes = await response.read()
                    base64_photo = base64.b64encode(file_bytes).decode('utf-8')

                    if not base64_photo:
                        logger.info('No photo in message')

                        return {
                            'err': 'Error while downloading photo',
                            'ok': None,
                        }

                    request.image = base64_photo

        return {
            'ok': self.client.upscale(request).to_dict(),
            'err': None,
        }

    @activity.defn
    async def _novita_progress(self, task_id: str) -> dict:
        return {
            'ok': self.client.progress(task_id).to_dict(),
            'err': None,
        }

    @activity.defn
    async def _reimagine(self, file_url: str) -> dict:
        image = await ImageUtil.download_and_open_file(file_url)

        if image.is_err():
            return {
                'err': image.err(),
                'ok': None,
            }

        image = image.ok()

        if image.width > 1024 or image.height > 1024:
            if image.width > image.height:
                # The width is larger than the height
                aspect_ratio = image.height / image.width
                new_width = 1024
                new_height = int(aspect_ratio * new_width)
            else:
                aspect_ratio = image.width / image.height
                new_height = 1024
                new_width = int(aspect_ratio * new_height)

            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert the (possibly) resized image back into bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        byte_data = buffer.getvalue()

        # Step 4: Encode the image back to base64
        base64_photo = base64.b64encode(byte_data).decode('utf-8')

        if not base64_photo:
            logger.info('No photo in message')

            return {
                'err': 'Error while downloading photo',
                'ok': None,
            }

        try:
            response = self.client.reimagine(
                image=base64_photo,
                response_image_type='png',
            )
        except NovitaResponseError as e:
            logger.error(f"Reimagine failed: {e}")

            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response.to_dict(),
            'err': None,
        }

    @staticmethod
    def shrink_to_size(image: Image, width: int = 1024, height: int = 1024) -> Image:
        if image.width > width or image.height > height:
            if image.width > image.height:
                # The width is larger than the height
                aspect_ratio = image.height / image.width
                new_width = width
                new_height = int(aspect_ratio * new_width)
            else:
                aspect_ratio = image.width / image.height
                new_height = height
                new_width = int(aspect_ratio * new_height)

            image = image.resize((new_width, new_height), Image.LANCZOS)

        return image

    @staticmethod
    def enlarge_to_size(image: Image, width: int = 1024, height: int = 1024) -> Image:
        if image.width < width or image.height < height:
            if image.width < image.height:
                # The width is larger than the height
                aspect_ratio = image.height / image.width
                new_width = width
                new_height = int(aspect_ratio * new_width)
            else:
                aspect_ratio = image.width / image.height
                new_height = height
                new_width = int(aspect_ratio * new_height)

            image = image.resize((new_width, new_height), Image.LANCZOS)

        return image

    @staticmethod
    def place_image_on_size(
            image: Image,
            width: int = 1024,
            height: int = 1024,
            position_x: int = None,
            position_y: int = None,
    ) -> Image:
        # Calculate the target size while maintaining the aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = min(width, image.width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, image.height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create a transparent image with the specified dimensions
        background = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Set default positions if None
        if position_x is None:
            position_x = (width - new_width) // 2
        if position_y is None:
            position_y = (height - new_height) // 2

        # Adjust positions to ensure the image stays within the canvas
        position_x = max(-new_width, min(position_x, width))
        position_y = max(-new_height, min(position_y, height))

        # Paste the resized image onto the transparent background
        background.paste(resized_image, (position_x, position_y), resized_image)

        return background

    @staticmethod
    def image_to_base64(image: Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        byte_data = buffer.getvalue()

        return base64.b64encode(byte_data).decode('utf-8')


    @activity.defn
    async def _swap_faces(self, image_url: str, face_image_url: str) -> dict:
        image = await ImageUtil.download_and_open_file(image_url)

        if image.is_err():
            return {
                'err': image.err(),
                'ok': None,
            }

        image = image.ok()

        image = self.shrink_to_size(image)
        base64_photo = self.image_to_base64(image)

        if not base64_photo:
            logger.info('No photo in message')

            return {
                'err': 'Error while downloading original photo',
                'ok': None,
            }

        try:
            face_img = await ImageUtil.download_and_open_file(face_image_url)
            if face_img.is_err():
                return {
                    'err': face_img.err(),
                    'ok': None,
                }

            face_image = self.shrink_to_size(face_img.ok())
            base64_face_photo = self.image_to_base64(face_image)
        except UnidentifiedImageError:
            return {
                'err': 'Данный формат изображения не поддерживается',
                'ok': None,
            }

        if not base64_face_photo:
            logger.info('No photo in message')

            return {
                'err': 'Error while downloading face photo',
                'ok': None,
            }

        try:
            response = self.client.merge_face(
                image=base64_photo,
                face_image=base64_face_photo,
                response_image_type='jpeg',
            )
        except NovitaResponseError as e:
            logger.error(f"Face swap failed: {e}")

            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response.to_dict(),
            'err': None,
        }

    @staticmethod
    async def swap_faces(image_url: str, face_image_url: str) -> Result[MergeFaceResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._swap_faces,
            args=[image_url, face_image_url],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(MergeFaceResponse.from_dict(result['ok']))

    @staticmethod
    async def upscale(request: UpscaleRequest, file_url: Optional[str] = None) -> Result[UpscaleResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._upscale,
            args=[request.to_dict(), file_url],
            start_to_close_timeout=timedelta(seconds=30)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(UpscaleResponse.from_dict(result['ok']))

    @staticmethod
    async def progress(task_id: str) -> Result[ProgressResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._novita_progress,
            args=[task_id],
            start_to_close_timeout=timedelta(seconds=30)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(ProgressResponse.from_dict(result['ok']))

    @staticmethod
    async def wait_for_task(task_id, wait_for=600) -> Result[ProgressResponse, str]:
        i = 0

        while i < wait_for:
            logger.info(f"Waiting for task {task_id} to complete")

            progress_info = await NovitaClient.progress(task_id)

            if progress_info.is_err():
                return Err(progress_info.err())

            progress = progress_info.ok()

            logger.info(f"Task {task_id} progress eta_relative: {progress.data.eta_relative}")

            if progress.data.status.finished():
                logger.info(f"Task {task_id} completed")

                return Ok(progress)

            await asyncio.sleep(1)
            i += 1

        return Err(f"Task {task_id} failed to complete in {wait_for} seconds")

    @staticmethod
    async def upscale_sync(request: UpscaleRequest, file_url: Optional[str] = None) -> Result[ProgressResponse, str]:
        response = await NovitaClient.upscale(request, file_url)

        if response.is_err():
            return Err(response.err())

        result = response.ok()

        if result.data is None:
            return Err(f"Upscale failed with result {result.msg}, code: {result.code}")

        return await NovitaClient.wait_for_task(result.data.task_id)

    @activity.defn
    async def _remove_background(self, file_url: str, skip_if_transparent: bool = False) -> dict:
        image = await ImageUtil.download_and_open_file(file_url)

        if image.is_err():
            return {
                'err': image.err(),
                'ok': None,
            }

        image = image.ok()

        try:
            alpha_channel = image.getchannel('A')
        except ValueError:
            alpha_channel = None

        has_background = not bool(alpha_channel and alpha_channel.getbbox())

        image = self.shrink_to_size(image, 1024, 1024)
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        base64_photo = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if not base64_photo:
            logger.info('No photo in message')

            return {
                'err': 'Error while downloading photo',
                'ok': None,
            }

        if not has_background and skip_if_transparent:
            logger.info("Image has no background")

            return {
                'err': None,
                'ok': base64_photo,
            }

        request = RemoveBackgroundRequest(
            image_file=base64_photo,
            extra={
                'response_image_type': 'png',
            }
        )

        response = RemoveBackgroundResponse.from_dict(
            self.client._post('/v3/remove-background', request.to_dict())
        )

        img = Image.open(BytesIO(base64.b64decode(response.image_file)))
        img = self.shrink_to_size(img, image.width, image.height)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_photo = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'ok': base64_photo,
            'err': None,
        }

    @activity.defn
    async def _place_image_on_size_activity(self, base64_img: str, width: int, height: int) -> str:
        image = ImageUtil.open_file(base64.b64decode(base64_img))

        if image.is_err():
            raise image.err()

        image = image.ok()
        image = self.place_image_on_size(image, width, height)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_img

    @activity.defn
    async def _replace_background(self, file_url: str, prompt: str) -> dict:
        image = await ImageUtil.download_and_open_file(file_url)

        if image.is_err():
            return {
                'err': image.err(),
                'ok': None,
            }

        image = image.ok()

        image = self.shrink_to_size(image, 1024, 1024)
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        base64_photo = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if not base64_photo:
            logger.info('No photo in message')

            return {
                'err': 'Error while downloading photo',
                'ok': None,
            }

        response = self.client.replace_background(
            image=base64_photo,
            prompt=prompt,
            response_image_type='jpeg',
        )

        img = Image.open(BytesIO(base64.b64decode(response.image_file)))
        img = self.shrink_to_size(img, image.width, image.height)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        base64_photo = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'ok': base64_photo,
            'err': None,
        }

    @activity.defn
    async def _upload_assets(self, images: list) -> dict:
        try:
            return {
                'ok': self.client.upload_assets(images),
                'err': None
            }
        except Exception as e:
            return {
                'ok': None,
                'err': str(e)
            }

    @staticmethod
    async def upload_assets(images: list) -> Result[List[str], str]:
        result = await workflow.execute_activity(
            NovitaClient._upload_assets,
            args=[images],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(result['ok'])

    @activity.defn
    async def _create_dreambooth_task(
        self,
        request_dict: dict,
    ) -> dict:
        try:
            response = self.client._post('/v3/async/make-photo', request_dict)

            return {
                'ok': response,
                'err': None,
            }
        except Exception as e:
            return {
                'ok': None,
                'err': str(e),
            }

    @staticmethod
    async def create_dreambooth_task(request: MakePhotoRequest) -> Result[str, str]:
        result = await workflow.execute_activity(
            NovitaClient._create_dreambooth_task,
            args=[request.to_dict()],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(MakePhotoResponse.from_dict(result['ok']).task_id)

    @activity.defn
    async def _create_text2img_task(self, request: dict, controlnet_units_images: Optional[dict[str]] = None) -> dict:
        request = Txt2ImgRequest.from_dict(request)
        if controlnet_units_images is not None:
            controlnet_units_images_keys = controlnet_units_images.keys()
            for controlnet_unit in request.controlnet_units:
                if controlnet_unit.input_image and controlnet_unit.input_image in controlnet_units_images_keys:
                    controlnet_unit.input_image = controlnet_units_images[controlnet_unit.input_image]

                if controlnet_unit.mask and controlnet_unit.mask in controlnet_units_images_keys:
                    controlnet_unit.mask = controlnet_units_images[controlnet_unit.mask]

        if request.seed == -1:
            request.seed = None

        try:
            response = self.client._post('/v2/txt2img', request.to_dict())
        except Exception as e:
            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response,
            'err': None,
        }

    @staticmethod
    async def create_text2img_task(
        request: Txt2ImgRequest,
        controlnet_units_images: Optional[dict[str]] = None,
    ) -> Result[Txt2ImgResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._create_text2img_task,
            args=[request.to_dict(), controlnet_units_images],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(Txt2ImgResponse.from_dict(result['ok']))

    @staticmethod
    async def replace_background(
        file_url: str,
        prompt: str,
    ) -> Result[Image, str]:
        result = await workflow.execute_activity(
            NovitaClient._replace_background,
            args=[file_url, prompt],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(result['ok'])

    @staticmethod
    async def place_image_on_size_activity(base64_img: str, width: int, height: int) -> str:
        return await workflow.execute_activity(
            NovitaClient._place_image_on_size_activity,
            args=[base64_img, width, height],
            start_to_close_timeout=timedelta(seconds=60)
        )

    @staticmethod
    async def remove_background(file_url: str, skip_if_transparent: bool = False) -> Result[str, str]:
        result = await workflow.execute_activity(
            NovitaClient._remove_background,
            args=[file_url, skip_if_transparent],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(result['ok'])

    @staticmethod
    async def reimagine(file_url: str) -> Result[ReimagineResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._reimagine,
            args=[file_url],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(ReimagineResponse.from_dict(result['ok']))

    @activity.defn
    async def _task_progress_v2(self, task_id: str) -> dict:
        try:
            response = self.client._get('/v2/progress', {
                'task_id': task_id,
            })
            logger.info(response)
        except Exception as e:
            logger.error(f"Failed to get task progress: {e}")

            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response,
            'err': None,
        }

    @staticmethod
    async def task_progress_v2(task_id: str) -> Result[ProgressResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._task_progress_v2,
            args=[task_id],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(ProgressResponse.from_dict(result['ok']))

    @activity.defn
    async def _task_result_v3(self, task_id: str) -> dict:
        try:
            response = self.client._get('/v3/async/task-result', {
                'task_id': task_id,
            })
        except Exception as e:
            logger.error(f"Failed to get task result: {e}")

            return {
                'err': str(e),
                'ok': None,
            }

        return {
            'ok': response,
            'err': None,
        }

    @staticmethod
    async def task_result_v3(task_id: str) -> Result[V3TaskResponse, str]:
        result = await workflow.execute_activity(
            NovitaClient._task_result_v3,
            args=[task_id],
            start_to_close_timeout=timedelta(seconds=60)
        )

        if result['err']:
            return Err(result['err'])

        return Ok(V3TaskResponse.from_dict(result['ok']))
