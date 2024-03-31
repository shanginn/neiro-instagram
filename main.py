import asyncio
import logging
import os
import random

import uuid
from dotenv import load_dotenv
from temporalio.client import Client

from CreatePostWorkflow import CreatePostWorkflow, CreatePostWorkflowInput

load_dotenv(os.getenv('ENV_FILE', '.env'))

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


temporal_address = os.getenv('TEMPORAL_ADDRESS')
task_queue = os.getenv('TASK_QUEUE')

async def main():
    client = await Client.connect(temporal_address)

    handle = await client.start_workflow(
        workflow=CreatePostWorkflow.run,
        args=[CreatePostWorkflowInput()],
        id=uuid.uuid4().hex,
        task_queue=task_queue,
    )

    print(f"Started workflow with ID: {handle.id}")

    result = await handle.result()

    print(f"Workflow completed with result: {result}")

    return result


if __name__ == "__main__":
    asyncio.run(main())