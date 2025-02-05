from typing import Any, Dict, Optional

import aiohttp

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AvatarQuality(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class VoiceEmotion(str, Enum):
    EXCITED = "excited"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    SOOTHING = "soothing"
    BROADCASTER = "broadcaster"


class ElevenLabsSettings(BaseModel):
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    style: Optional[int] = None
    use_speaker_boost: Optional[bool] = None


class VoiceSettings(BaseModel):
    voice_id: Optional[str] = Field(None, alias="voiceId")
    rate: Optional[float] = None
    emotion: Optional[VoiceEmotion] = None
    elevenlabs_settings: Optional[ElevenLabsSettings] = Field(
        None, alias="elevenlabsSettings"
    )


class NewSessionRequest(BaseModel):
    avatarName: str
    quality: Optional[AvatarQuality] = None
    knowledgeId: Optional[str] = None
    knowledgeBase: Optional[str] = None
    voice: Optional[VoiceSettings] = None
    language: Optional[str] = None
    version: Literal["v2"] = "v2"
    video_encoding: Literal["H264"] = "H264"
    source: Literal["sdk"] = "sdk"
    disableIdleTimeout: Optional[bool] = None


class SessionResponse(BaseModel):
    session_id: str
    access_token: str
    realtime_endpoint: str
    url: str


class HeygenAPIError(Exception):
    """Custom exception for HeyGen API errors."""

    def __init__(self, message: str, status: int, response_text: str) -> None:
        super().__init__(message)
        self.status = status
        self.response_text = response_text


class HeyGenClient:
    """HeyGen Streaming API client."""

    BASE_URL = "https://api.heygen.com/v1"

    def __init__(
        self, api_key: str, session: Optional[aiohttp.ClientSession] = None
    ) -> None:
        self.api_key = api_key
        self.session = session or aiohttp.ClientSession()

    async def request(self, path: str, params: Dict[str, Any]) -> Any:
        """
        Make a POST request to the HeyGen API.

        Args:
            path (str): API endpoint path.
            params (Dict[str, Any]): JSON-serializable parameters.

        Returns:
            Any: Parsed JSON response data.

        Raises:
            APIError: If the API response is not successful.
        """
        url = f"{self.BASE_URL}{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        async with self.session.post(url, json=params, headers=headers) as response:
            if not response.ok:
                response_text = await response.text()
                print("heygen api error", response_text)
                raise HeygenAPIError(
                    f"API request failed with status {response.status}",
                    response.status,
                    response_text,
                )
            json_data = await response.json()
            return json_data.get("data")

    async def new_session(self, request_data: NewSessionRequest) -> SessionResponse:
        params = {
            "avatar_name": request_data.avatarName,
            "quality": request_data.quality,
            "knowledge_base_id": request_data.knowledgeId,
            "knowledge_base": request_data.knowledgeBase,
            "voice": {
                "voice_id": request_data.voice.voiceId if request_data.voice else None,
                "rate": request_data.voice.rate if request_data.voice else None,
                "emotion": request_data.voice.emotion if request_data.voice else None,
                "elevenlabs_settings": (
                    request_data.voice.elevenlabsSettings
                    if request_data.voice
                    else None
                ),
            },
            "language": request_data.language,
            "version": "v2",
            "video_encoding": "H264",
            "source": "sdk",
            "disable_idle_timeout": request_data.disableIdleTimeout,
        }
        session_info = await self.request("/streaming.new", params)
        print("heygen session info", session_info)

        return SessionResponse.model_validate(session_info)

    async def start_session(self, session_id: str) -> Any:
        """
        Start the streaming session.

        Returns:
            Any: Response data from the start session API call.
        """
        if not session_id:
            raise ValueError("Session ID is not set. Call new_session first.")

        params = {
            "session_id": session_id,
        }
        return await self.request("/streaming.start", params)

    async def close(self) -> None:
        """Close the aiohttp session."""
        await self.session.close()
