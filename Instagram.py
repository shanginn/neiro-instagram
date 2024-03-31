from datetime import timedelta
from pathlib import Path
from typing import List, Dict

from instagrapi import Client
from instagrapi.types import Usertag, Location, Media
from temporalio import activity, workflow

from ImageUtil import ImageUtil
import tempfile

class Instagram:
    def __init__(self, username: str, password: str, session_file: str = 'instagram_session.json'):
        self.username = username
        self.password = password
        self.client: Client = Client()
        self.session_file = Path(session_file)

    def login(self, relogin: bool = False):
        if not relogin and self.session_file.exists():
            self.client.load_settings(self.session_file)
            return

        self.client.login(self.username, self.password, relogin=relogin)
        self.client.dump_settings(self.session_file)

    def client(self):
        return self.client

    def get_activities(self) -> List:
        return [
            self._photo_upload,
        ]

    @activity.defn
    async def _photo_upload(
        self,
        photo: str,
        caption: str,
        upload_id: str = "",
        usertags: List[Usertag] = [],
        location: Location = None,
        extra_data: Dict[str, str] = {},
    ) -> dict:
        tmp_file = tempfile.NamedTemporaryFile()
        path = Path(tmp_file.name + '.png')

        await ImageUtil.download_and_save_file(photo, path)

        media = self.client.photo_upload(
            path,
            caption,
            upload_id,
            usertags,
            location,
            extra_data,
        )

        return {
            'id': media.id,
            'code': media.code,
            'caption_text': media.caption_text,
            'title': media.title,
        }

    @staticmethod
    async def photo_upload(
        photo: str,
        caption: str,
        upload_id: str = "",
        usertags: List[Usertag] = [],
        location: Location = None,
        extra_data: Dict[str, str] = {},
    ) -> dict:
        return await workflow.execute_activity(
            Instagram._photo_upload,
            args=[photo, caption, upload_id, usertags, location, extra_data],
            start_to_close_timeout=timedelta(seconds=60)
        )
