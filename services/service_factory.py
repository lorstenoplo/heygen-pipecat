"""
Service factory for creating and configuring pipeline services.
"""

import os
from config.settings import settings
from heygen import HeyGenVideoService
from heygen_client import AvatarQuality, HeyGenClient, NewSessionRequest
from pipecat.services.openai import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions, DeepgramTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from prompts.system_prompts import KATYA_SYSTEM_PROMPT
from loguru import logger


class ServiceFactory:
    """Factory for creating pipeline services."""
    
    @staticmethod
    def create_transport(room: str, token: str) -> DailyTransport:
        """Create Daily transport with configured parameters."""
        return DailyTransport(
            room,
            token,
            "HeyGen",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1280,
                camera_out_height=1120,
                vad_enabled=True,
                transcription_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.0)),
                vad_audio_passthrough=True,
                video_out_enabled=True,
                video_in_enabled=True,
            ),
        )
    
    @staticmethod
    def create_stt_service() -> DeepgramSTTService:
        """Create Deepgram STT service."""
        return DeepgramSTTService(
            api_key=settings.DEEPGRAM_API_KEY,
            live_options=LiveOptions(language="en-US"),
        )
    
    @staticmethod
    def create_tts_service() -> ElevenLabsTTSService:
        """Create ElevenLabs TTS service."""
        return ElevenLabsTTSService(
            api_key=settings.ELEVENLABS_API_KEY, 
            voice_id="21m00Tcm4TlvDq8ikWAM"
        )
    
    @staticmethod
    def create_llm_service() -> OpenAILLMService:
        """Create OpenAI LLM service with vision support."""        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="gpt-4o",
        )
        
        # Register scheduling functions using new direct method
        from tools.scheduling_tools import show_scheduling_popup
        llm.register_direct_function(show_scheduling_popup)
        
        return llm
    
    @staticmethod
    def create_llm_context() -> OpenAILLMContext:
        """Create LLM context with system prompt and tools."""
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        
        # Define scheduling tools schema
        show_popup_function = FunctionSchema(
            name="show_scheduling_popup",
            description="Opens a scheduling popup when user wants to book an appointment. Use this when user wants to schedule anything. DO NOT ask for date/time/email through conversation - just call this tool.",
            properties={
                "service_type": {
                    "type": "string",
                    "enum": ["consultation", "demo", "onboarding", "support", "training"],
                    "description": "Type of service to schedule"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about what's being scheduled"
                }
            },
            required=["service_type"]
        )
        
        tools = ToolsSchema(standard_tools=[show_popup_function])
        
        messages = [
            {
                "role": "system",
                "content": KATYA_SYSTEM_PROMPT,
            },
        ]
        return OpenAILLMContext(messages, tools)  # type: ignore
    
    @staticmethod
    def create_rtvi_processor() -> RTVIProcessor:
        """Create RTVI processor."""
        return RTVIProcessor(config=RTVIConfig(config=[]))
    
    @staticmethod
    def create_transcript_processor() -> TranscriptProcessor:
        """Create transcript processor."""
        return TranscriptProcessor()
    
    @staticmethod
    async def create_heygen_client(session) -> HeyGenClient:
        """Create HeyGen client."""
        return HeyGenClient(api_key=settings.HEYGEN_API_KEY, session=session)
    
    @staticmethod
    async def create_heygen_session(heygen_client: HeyGenClient):
        """Create and start HeyGen session."""
        try:
            session_response = await heygen_client.new_session(
                NewSessionRequest(
                    avatarName="Katya_Chair_Sitting_public",
                    version="v1",
                    quality=AvatarQuality.high,
                )
            )
            logger.info(f"HeyGen session created: {session_response.session_id}")
            
            # Start session
            await heygen_client.start_session(session_response.session_id)
            logger.info(f"HeyGen session started: {session_response.session_id}")
            
            return session_response
        except Exception as e:
            logger.error(f"Failed to create/start HeyGen session: {e}")
            raise
    
    @staticmethod
    async def create_heygen_video_service(session_response, session) -> HeyGenVideoService:
        """Create HeyGen video service."""
        try:
            return HeyGenVideoService(
                session_id=session_response.session_id,
                session_token=session_response.access_token,
                session=session,
                realtime_endpoint=session_response.realtime_endpoint,
                livekit_room_url=session_response.url,
            )
        except Exception as e:
            logger.error(f"Failed to create HeyGen video service: {e}")
            raise
