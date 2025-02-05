from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class AppSettings(BaseSettings):
    HEYGEN_API_KEY: str
    DAILYCO_API_KEY: str
    DAILYCO_BASE_URL: str
    DAILYCO_ROOM_BASE_URL: str
    ELEVENLABS_API_KEY: str
    OPENAI_API_KEY: str
    DEEPGRAM_API_KEY: str

    model_config = SettingsConfigDict()


settings = AppSettings()
