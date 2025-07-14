from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


load_dotenv()


class AppSettings(BaseSettings):
    HEYGEN_API_KEY: str = ""
    DAILY_API_KEY: str = ""
    DAILY_API_URL: str = ""
    DAILY_API_ROOM_BASE_URL: str = ""
    ELEVENLABS_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    RESEND_API_KEY: str = ""  # Optional for scheduling feature

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = AppSettings(
    HEYGEN_API_KEY=os.getenv("HEYGEN_API_KEY", ""),
    DAILY_API_KEY=os.getenv("DAILY_API_KEY", ""),
    DAILY_API_URL=os.getenv("DAILY_API_URL", ""),
    DAILY_API_ROOM_BASE_URL=os.getenv("DAILY_API_ROOM_BASE_URL", ""),
    ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY", ""),
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
    DEEPGRAM_API_KEY=os.getenv("DEEPGRAM_API_KEY", ""),
    RESEND_API_KEY=os.getenv("RESEND_API_KEY", "")
)
