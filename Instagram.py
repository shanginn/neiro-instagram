import random
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

        if location is None:
            random_cities = [
                (36.8979091, 30.6357046),  # Antalya, Turkey
                (41.004852, 28.6825454),  # Istanbul, Turkey
                (55.755825, 37.617298),  # Moscow, Russia
                (51.5073219, -0.1276474),  # London, United Kingdom
                (40.7127281, -74.0060152),  # New York City, USA
                (34.052235, -118.243683),  # Los Angeles, USA
                (51.507322, -0.127647),  # London, UK
                (35.689487, 139.691711),  # Tokyo, Japan
                (40.712776, -74.005974),  # New York City, USA
                (48.856613, 2.352222),  # Paris, France
                (19.432608, -99.133209),  # Mexico City, Mexico
                (55.755825, 37.617298),  # Moscow, Russia
                (39.904202, 116.407394),  # Beijing, China
                (28.613939, 77.209023),  # New Delhi, India
                (-34.603722, -58.381592),  # Buenos Aires, Argentina
                (30.044420, 31.235712),  # Cairo, Egypt
                (-23.550520, -46.633308),  # São Paulo, Brazil
                (41.008240, 28.978359),  # Istanbul, Turkey
                (33.868820, 151.209290),  # Sydney, Australia
                (37.774929, -122.419418),  # San Francisco, USA
                (52.520008, 13.404954),  # Berlin, Germany
                (23.129110, 113.264385),  # Guangzhou, China
                (1.352083, 103.819839),  # Singapore
                (35.689500, 51.388973),  # Tehran, Iran
                (22.572645, 88.363892)  # Kolkata, India
            ]

            variation = 0.001
            random_city = random.choice(random_cities)

            lat, lon = random_city
            new_lat = lat + random.uniform(-variation, variation)
            new_lon = lon + random.uniform(-variation, variation)

            location = self.client.location_search(lat=new_lat, lng=new_lon)[0]

        try:
            media = self.client.photo_upload(
                path,
                caption,
                upload_id,
                usertags,
                location,
                extra_data,
            )
        except Exception as e:
            return {
                'ok': None,
                'err': str(e),
            }

        return {
            'ok': {
                'id': media.id,
                'code': media.code,
                'caption_text': media.caption_text,
                'title': media.title,
            },
            'err': None
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
