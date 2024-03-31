import random
from datetime import timedelta

from temporalio import activity, workflow


class Util:
    def get_activities(self):
        return [
            self._seed
        ]

    @activity.defn
    async def _seed(self) -> int:
        return random.randrange(0, 2 ** 32 + 1)

    @staticmethod
    async def get_seed() -> int:
        return await workflow.execute_activity(
            Util._seed,
            start_to_close_timeout=timedelta(seconds=5)
        )
