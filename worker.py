import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker as TemporalWorker

from CreatePostWorkflow import CreatePostWorkflow
from Instagram import Instagram
from Novita import NovitaClient
from PhotoPromptGenerator import PhotoPromptGenerator
from Replicate import Replicate
from Util import Util

load_dotenv(os.getenv('ENV_FILE', '.env'))

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


temporal_address = os.getenv('TEMPORAL_ADDRESS')

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
omni_api_key = os.getenv('NOVITA_API_KEY')
replicate_api_token = os.getenv('REPLICATE_API_TOKEN')

instagram_creds = (os.getenv('INSTAGRAM_USERNAME'), os.getenv('INSTAGRAM_PASSWORD'))

task_queue = os.getenv('TASK_QUEUE')


class Worker:
    async def run(self):
        logger.info("Starting worker")

        client = await Client.connect(temporal_address)
        novita = NovitaClient(omni_api_key)
        replicate = Replicate(replicate_api_token)

        instagram = Instagram(*instagram_creds)
        instagram.login()

        photo_prompt_generator = PhotoPromptGenerator(anthropic_api_key)
        util = Util()

        logger.info("Worker started")

        await TemporalWorker(
            client,
            task_queue=task_queue,
            workflows=[
                CreatePostWorkflow,
            ],
            activities=[
                *novita.get_activities(),
                *instagram.get_activities(),
                *photo_prompt_generator.get_activities(),
                *util.get_activities(),
                *replicate.get_activities(),
            ]
        ).run()


if __name__ == "__main__":
    logger.info("Initializing worker...")

    worker = Worker()

    logger.info("Worker initialized. Running.")
    asyncio.run(worker.run())
