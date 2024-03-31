import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path

from instagrapi import Client

from dotenv import load_dotenv

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()

instagram_username = os.getenv('INSTAGRAM_USERNAME')
instagram_password = os.getenv('INSTAGRAM_PASSWORD')

# Database setup
conn = sqlite3.connect('uploads.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()


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

async def main():
    insta = Instagram(
        instagram_username,
        instagram_password,
        'instagram_session.json'
    )

    insta.login()

    photos_path = Path('photos')
    for photo in photos_path.iterdir():
        if photo.is_file() and photo.suffix in ['.jpg', '.jpeg', '.png']:
            # Check if the photo has been uploaded before
            cursor.execute('SELECT * FROM uploads WHERE filename = ?', (str(photo),))
            if cursor.fetchone() is None:
                # Upload the photo
                media = insta.client.photo_upload(
                    photo,
                    caption=""
                )

                # Log the upload in the database
                cursor.execute('INSERT INTO uploads (filename) VALUES (?)', (str(photo),))
                conn.commit()
                logger.info(f"Uploaded {photo.name}")

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    asyncio.run(main())
