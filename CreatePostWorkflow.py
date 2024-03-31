import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Awaitable, List, Callable, Coroutine, Any
from result import Result, Ok, Err
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from PhotoPromptGenerator import PhotoPromptGenerator
    from Util import Util
    from Novita import NovitaClient
    from Instagram import Instagram
    from novita_client import Samplers, Txt2ImgRequest, Refiner, ProgressResponse, ProgressResponseStatusCode, \
        ProgressData


@dataclass
class CreatePostWorkflowInput:
    width: int = 1024
    height: int = 1024


@workflow.defn
class CreatePostWorkflow:
    data: CreatePostWorkflowInput

    @workflow.run
    async def run(self, data: CreatePostWorkflowInput):
        workflow.logger.info(f'Executing workflow with data: {data}')

        prompt = await PhotoPromptGenerator.generate_photo_prompt()

        workflow.logger.info(f'Generated prompt: {prompt}')

        negative_prompt = 'worst quality, low resolution, bad hands, distorted, twisted, watermark, blurry'
        seed = await Util.get_seed()

        request = Txt2ImgRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=1,
            model_name='leosamsHelloworldSDXL_helloworldSDXL50_268813.safetensors',
            sampler_name=Samplers.DPMPP_M_KARRAS,
            width=data.width,
            height=data.height,
            steps=25,
            seed=seed,
            sd_refiner=Refiner(
                checkpoint='sd_xl_refiner_1.0.safetensors',
                switch_at=0.75,
            ),
        )

        result = await self.generate_images(request)

        if result.is_err():
            workflow.logger.error(f'Error generating images: {result.err()}')
            return

        images = result.unwrap().imgs
        # swapped_images = []
        #
        # for photo in images:
        #     swapped_photo_result = await NovitaClient.swap_faces(
        #         photo,
        #         str(Path('photos/face.png').absolute()),
        #     )
        #
        #     if swapped_photo_result.is_err():
        #         workflow.logger.error(f'Error swapping faces: {swapped_photo_result.err()}')
        #         swapped_images.append(photo)
        #         continue
        #
        #     swapped_images.append(swapped_photo_result.ok().image_file)
        #
        #
        # return swapped_images

        await Instagram.photo_upload(
            photo=images[0],
            caption=''
        )

        return images

    async def generate_images(
        self,
        request: Txt2ImgRequest,
        send_intermediate_images: Optional[Callable] = None,
    ) -> Result[ProgressData, str]:
        response = await NovitaClient.create_text2img_task(request)

        workflow.logger.info(f'Generator text2img response: {response}')

        if response.is_err():
            return response

        task_id = response.unwrap().data.task_id

        task_data = await self.wait_for_task(task_id, send_intermediate_images)

        workflow.logger.info(f'Task data: {task_data}')

        if task_data.is_err():
            return task_data

        generation_data = task_data.ok()

        if generation_data.imgs is None:
            return Err(f"Task failed, no images")

        return Ok(generation_data)

    async def wait_for_task(
        self,
        task_id,
        send_intermediate_images: Optional[Callable[[List[str], str], Awaitable[None]]] = None,
    ) -> Result[ProgressData, str]:
        await asyncio.sleep(5)

        while True:
            progress_response = await NovitaClient.task_progress_v2(task_id)

            if progress_response.is_err():
                return progress_response

            progress = progress_response.ok().data
            task_failed = (progress.status == ProgressResponseStatusCode.FAILED
                           or progress.status == ProgressResponseStatusCode.TIMEOUT)

            workflow.logger.info(f"Task {task_id} progress eta_relative: {progress.eta_relative}")

            if progress.status == ProgressResponseStatusCode.RUNNING:
                if send_intermediate_images:
                    current_images = [image for image in progress.current_images if image]

                    if current_images:
                        await send_intermediate_images(
                            current_images,
                            f'Генерация в процессе, это промежуточный результат...'
                        )
            elif progress.status == ProgressResponseStatusCode.SUCCESSFUL:
                workflow.logger.info(f"Task {task_id} completed")

                return Ok(progress)
            elif task_failed:
                workflow.logger.error(f"Task {task_id} failed, reason: {progress.failed_reason}")

                return Err(progress.failed_reason)

            await asyncio.sleep(2)
