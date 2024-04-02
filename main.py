import asyncio
import logging
import os
import random

import uuid
from datetime import timedelta
from typing import Any, Coroutine

from dotenv import load_dotenv
from temporalio.client import Client, Schedule, ScheduleActionStartWorkflow, ScheduleSpec, ScheduleIntervalSpec, \
    ScheduleHandle

from CreatePostWorkflow import CreatePostWorkflow, CreatePostWorkflowInput

load_dotenv(os.getenv('ENV_FILE', '.env'))

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


temporal_address = os.getenv('TEMPORAL_ADDRESS')
task_queue = os.getenv('TASK_QUEUE')

async def run_workflow(client: Client):
    handle = await client.start_workflow(
        workflow=CreatePostWorkflow.run,
        args=[CreatePostWorkflowInput()],
        id=uuid.uuid4().hex,
        task_queue=task_queue,
    )

    print(f"Started workflow with ID: {handle.id}")

    return handle


async def start_schedule(client: Client, interval: timedelta) -> ScheduleHandle:
    return await client.create_schedule(
        f"Каждые {interval} [{uuid.uuid4().hex}]",
        Schedule(
            action=ScheduleActionStartWorkflow(
                CreatePostWorkflow.run,
                args=[CreatePostWorkflowInput()],
                id='Пост фотографии в инстаграм',
                task_queue=task_queue,
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=interval)],
                jitter=timedelta(minutes=5),
            ),
        ),
    )


async def main():
    client = await Client.connect(temporal_address)

    await run_workflow(client)


if __name__ == "__main__":
    asyncio.run(main())
