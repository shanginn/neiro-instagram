import os
from io import BytesIO
from pathlib import Path

import aiohttp
from PIL import Image, UnidentifiedImageError
from result import Result, Err, Ok


class ImageUtil:
    @staticmethod
    def open_file(image_bytes: bytes) -> Result[Image, str]:
        try:
            image = Image.open(BytesIO(image_bytes))
        except UnidentifiedImageError:
            return Err('Данный формат изображения не поддерживается')

        return Ok(image)

    @staticmethod
    async def download_and_open_file(file_url: str) -> Result[Image, str]:
        if os.path.isfile(file_url):
            return ImageUtil.open_file(
                open(file_url, 'rb').read()
            )
        else:
            return ImageUtil.open_file(
                await ImageUtil.download_file(file_url)
            )

    @staticmethod
    async def download_file(file_url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                return await response.read()

    @staticmethod
    async def download_and_save_file(file_url: str, file_path: Path) -> Result[Image, str]:
        image = await ImageUtil.download_and_open_file(file_url)

        if image.is_err():
            return image

        img: Image = image.ok()

        img.save(file_path)

        return Ok(img)
