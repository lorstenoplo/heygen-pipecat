# #
# # Copyright (c) 2024–2025, Daily
# #
# # SPDX-License-Identifier: BSD 2-Clause License
# #

# import asyncio
# import os
# import sys
# import webbrowser

# import aiohttp
# from dotenv import load_dotenv
# from loguru import logger
# from heygen import HeyGenVideoService
# from heygen_client import AvatarQuality, HeyGenClient, NewSessionRequest
# from runner import configure

# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineParams, PipelineTask
# from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
# from pipecat.services.openai import OpenAILLMService
# from pipecat.services.elevenlabs import ElevenLabsTTSService
# from pipecat.transports.services.daily import DailyParams, DailyTransport
# from config import settings
# from pipecat.services.deepgram import DeepgramSTTService, LiveOptions, DeepgramTTSService
# from pipecat.audio.vad.vad_analyzer import VADParams

# load_dotenv(override=True)

# logger.remove(0)
# logger.add(sys.stderr, level="DEBUG")


# async def main():
#     async with aiohttp.ClientSession() as session:
#         try:
#             room, token = await configure(session)
#             logger.info(f"Room configured: {room}")
            
#             # Open room URL in default browser
#             try:
#                 webbrowser.open(room)
#                 logger.info("Room URL opened in browser")
#             except Exception as e:
#                 logger.warning(f"Could not open room URL in browser: {e}")

#             transport = DailyTransport(
#                 room,
#                 token,
#                 "HeyGen",
#                 DailyParams(
#                     audio_out_enabled=True,
#                     camera_out_enabled=True,
#                     camera_out_width=1280,
#                     camera_out_height=1120,
#                     vad_enabled=True,
#                     transcription_enabled=True,
#                     audio_in_sample_rate=16000,
#                     audio_out_sample_rate=24000,
#                     vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
#                     vad_audio_passthrough=True,
#                     video_out_enabled=True,
#                 ),
#             )

#             stt = DeepgramSTTService(
#                 api_key=settings.DEEPGRAM_API_KEY,
#                 live_options=LiveOptions(language="en-US"),
#             )

#             tts = ElevenLabsTTSService(
#                 api_key=settings.ELEVENLABS_API_KEY, 
#                 voice_id="21m00Tcm4TlvDq8ikWAM"
#             )

#             # Create HeyGen client with better error handling
#             heygen_client = HeyGenClient(api_key=settings.HEYGEN_API_KEY, session=session)
#             logger.info("HeyGen client created")

#             # Create new session with error handling
#             try:
#                 from heygen_client import AvatarQuality
#                 session_response = await heygen_client.new_session(
#                     NewSessionRequest(
#                         avatarName="Katya_Chair_Sitting_public",
#                         version="v1",
#                         quality=AvatarQuality.high,
#                     )
#                 )
#                 logger.info(f"HeyGen session created: {session_response.session_id}")
#                 logger.info(f"Session URL: {session_response.url}")
#                 logger.info(f"Realtime endpoint: {session_response.realtime_endpoint}")
#             except Exception as e:
#                 logger.error(f"Failed to create HeyGen session: {e}")
#                 raise

#             # Start session with error handling
#             try:
#                 await heygen_client.start_session(session_response.session_id)
#                 logger.info(f"HeyGen session started: {session_response.session_id}")
#             except Exception as e:
#                 logger.error(f"Failed to start HeyGen session: {e}")
#                 raise

#             # Create video service with error handling
#             try:
#                 heygen_video_service = HeyGenVideoService(
#                     session_id=session_response.session_id,
#                     session_token=session_response.access_token,
#                     session=session,
#                     realtime_endpoint=session_response.realtime_endpoint,
#                     livekit_room_url=session_response.url,
#                 )
#                 logger.info("HeyGen video service created")
#             except Exception as e:
#                 logger.error(f"Failed to create HeyGen video service: {e}")
#                 raise

#             llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

#             messages = [
#                 {
#                     "role": "system",
#                     "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
#                 },
#             ]

#             context = OpenAILLMContext(messages) # type: ignore
#             context_aggregator = llm.create_context_aggregator(context)

#             pipeline = Pipeline(
#                 [
#                     transport.input(),
#                     stt,
#                     context_aggregator.user(),
#                     llm,
#                     tts,
#                     heygen_video_service,
#                     transport.output(),
#                     context_aggregator.assistant(),
#                 ]
#             )

#             task = PipelineTask(
#                 pipeline,
#                 params = PipelineParams(
#                     allow_interruptions=True,
#                     enable_usage_metrics=True,
#                 ),
#             )

#             @transport.event_handler("on_client_connected")
#             async def on_client_connected(transport, client):
#                 logger.info(f"Client connected")

#                 # Kick off the conversation.
#                 messages.append({"role": "system", "content": "Please introduce yourself to the user."})
#                 await task.queue_frames([context_aggregator.user().get_context_frame()])

#             @transport.event_handler("on_client_disconnected")
#             async def on_client_disconnected(transport, client):
#                 logger.info(f"Client disconnected")
#                 await task.cancel()

#             @transport.event_handler("on_first_participant_joined")
#             async def on_first_participant_joined(transport, participant):
#                 logger.info(f"First participant joined: {participant}")
#                 await transport.capture_participant_transcription(participant["id"])

#             @transport.event_handler("on_participant_left")
#             async def on_participant_left(transport, participant, reason):
#                 # stop the heygen session when the participant leaves
#                 try:
#                     await heygen_client.stop_session(session_response.session_id)
#                     logger.info(f"HeyGen session stopped for participant: {participant['id']}")
#                 except Exception as e:
#                     logger.error(f"Failed to stop HeyGen session: {e}")
                
#                 logger.info(f"Participant left: {participant}, reason: {reason}")

#             # Add more event handlers for debugging
#             @transport.event_handler("on_call_state_updated")
#             async def on_call_state_updated(transport, state):
#                 logger.info(f"Call state updated: {state}")

#             logger.info("Starting pipeline runner...")
#             runner = PipelineRunner()
#             await runner.run(task)

#         except Exception as e:
#             logger.error(f"Main execution error: {e}")
#             raise


# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import os
import sys
import webbrowser

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from heygen import HeyGenVideoService
from heygen_client import AvatarQuality, HeyGenClient, NewSessionRequest
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from config import settings
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.user_response import UserResponseAggregator
from pipecat.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.frames.frames import Frame, TextFrame, UserImageRequestFrame, LLMMessagesAppendFrame, UserImageRawFrame
from typing import Optional, Any

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

class UserImageRequester(FrameProcessor):
    """Custom processor to request user images when text is processed."""
    
    def __init__(self, participant_id: Optional[str] = None):
        super().__init__()
        self._participant_id = participant_id

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, TextFrame):
            await self.push_frame(
                UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
            )
        await self.push_frame(frame, direction)


def get_transport_client_id(transport: BaseTransport, client: Any) -> str:
    """Get client identifier from transport-specific client object.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.

    Returns:
        Client identifier string, empty if transport not supported.
    """
    if isinstance(transport, DailyTransport):
        return client["id"]
    logger.warning(f"Unable to get client id from unsupported transport {type(transport)}")
    return ""


async def maybe_capture_participant_camera(
    transport: BaseTransport, client: Any, framerate: int = 1
):
    """Capture participant camera video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 1 for better performance.
    """
    if isinstance(transport, DailyTransport):
        await transport.capture_participant_video(
            client["id"], framerate=framerate, video_source="camera"
        )

system_instructions = """
You are Katya, a friendly and engaging person who loves meeting new people. You have a warm, conversational personality and enjoy getting to know others through natural dialogue.

Your personality traits:
- Warm, friendly, and genuinely interested in others
- Curious about people's lives, interests, and experiences
- Great at making conversation flow naturally
- Empathetic and supportive
- Has a good sense of humor
- Remembers details from the conversation

Conversation style:
- Speak naturally like you're talking to a friend
- Ask follow-up questions to show genuine interest
- Share relatable thoughts and observations
- Use casual, friendly language
- Keep responses short and conversational (1-2 sentences usually)
- Respond quickly and naturally
- Don't mention that you're an AI or in a video call

You have excellent vision capabilities and can see the user through their camera feed. You can describe people, their appearance, clothing, expressions, and surroundings in detail. When you receive visual information, naturally describe what you observe in a friendly, conversational way. Feel free to comment on their appearance, what they're wearing, their surroundings, or anything interesting you notice.

If someone asks about what you can see, provide detailed descriptions and ask follow-up questions about what you're seeing.

Start conversations by introducing yourself as Katya and asking how their day is going. Be genuinely interested in getting to know them.
"""

async def main():
    async with aiohttp.ClientSession() as session:
        try:
            room, token = await configure(session)
            logger.info(f"Room configured: {room}")
            
            # Open room URL in default browser
            try:
                webbrowser.open(room)
                logger.info("Room URL opened in browser")
            except Exception as e:
                logger.warning(f"Could not open room URL in browser: {e}")

            transport = DailyTransport(
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
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                    vad_audio_passthrough=True,
                    video_out_enabled=True,
                    video_in_enabled=True,
                ),
            )

            user_response = UserResponseAggregator()

            # Initialize the image requester without setting the participant ID yet
            image_requester = UserImageRequester()

            # This aggregator will collect vision frames
            vision_aggregator = VisionImageFrameAggregator()

            stt = DeepgramSTTService(
                api_key=settings.DEEPGRAM_API_KEY,
                live_options=LiveOptions(language="en-US"),
            )

            tts = ElevenLabsTTSService(
                api_key=settings.ELEVENLABS_API_KEY, 
                voice_id="21m00Tcm4TlvDq8ikWAM"
            )

            # Create HeyGen client with better error handling
            heygen_client = HeyGenClient(api_key=settings.HEYGEN_API_KEY, session=session)
            logger.info("HeyGen client created")

            # Create new session with error handling
            try:
                from heygen_client import AvatarQuality
                session_response = await heygen_client.new_session(
                    NewSessionRequest(
                        avatarName="Katya_Chair_Sitting_public",
                        version="v1",
                        quality=AvatarQuality.high,
                    )
                )
                logger.info(f"HeyGen session created: {session_response.session_id}")
                logger.info(f"Session URL: {session_response.url}")
                logger.info(f"Realtime endpoint: {session_response.realtime_endpoint}")
            except Exception as e:
                logger.error(f"Failed to create HeyGen session: {e}")
                raise

            # Start session with error handling
            try:
                await heygen_client.start_session(session_response.session_id)
                logger.info(f"HeyGen session started: {session_response.session_id}")
            except Exception as e:
                logger.error(f"Failed to start HeyGen session: {e}")
                raise

            # Create video service with error handling
            try:
                heygen_video_service = HeyGenVideoService(
                    session_id=session_response.session_id,
                    session_token=session_response.access_token,
                    session=session,
                    realtime_endpoint=session_response.realtime_endpoint,
                    livekit_room_url=session_response.url,
                )
                logger.info("HeyGen video service created")
            except Exception as e:
                logger.error(f"Failed to create HeyGen video service: {e}")
                raise

            # Use GPT-4o model which supports vision
            llm = OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"), 
                model="gpt-4o"
            )

            messages = [
                {
                    "role": "system",
                    "content": system_instructions,
                },
            ]

            context = OpenAILLMContext(messages) # type: ignore
            context_aggregator = llm.create_context_aggregator(context)

            # Pipeline setup - CRITICAL: Order matters for vision to work
            pipeline = Pipeline(
                [
                    transport.input(),       # Input from transport
                    stt,                    # Speech to text
                    user_response,          # Aggregate user response
                    image_requester,        # Request user image
                    vision_aggregator,      # Aggregate vision frames - MUST be before context_aggregator.user()
                    context_aggregator.user(),  # Add user message + vision to context
                    llm,                    # LLM processes text + vision
                    tts,                    # Text to speech
                    heygen_video_service,   # Video output
                    transport.output(),     # Output to transport
                    context_aggregator.assistant(),  # Add assistant response to context
                ]
            )

            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_usage_metrics=True,
                ),
            )

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, client):
                logger.info(f"Client connected: {client}")

                # Capture participant camera with framerate for vision
                await maybe_capture_participant_camera(transport, client, framerate=1)

                # Set the participant ID in the image requester
                client_id = get_transport_client_id(transport, client)
                image_requester.set_participant_id(client_id)
                logger.info(f"Set participant ID: {client_id}")

                # Wait a moment for camera to initialize
                await asyncio.sleep(1)

                # Push a context frame to start the conversation
                await task.queue_frames(
                    [
                        LLMMessagesAppendFrame([
                            {"role":"user", "content": "Hello! Please introduce yourself and describe what you can see."}
                        ])
                    ]
                )

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.info(f"Client disconnected: {client}")
                await task.cancel()

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"First participant joined: {participant}")
                await transport.capture_participant_transcription(participant["id"])

            @transport.event_handler("on_participant_left")
            async def on_participant_left(transport, participant, reason):
                # stop the heygen session when the participant leaves
                try:
                    await heygen_client.stop_session(session_response.session_id)
                    logger.info(f"HeyGen session stopped for participant: {participant['id']}")
                except Exception as e:
                    logger.error(f"Failed to stop HeyGen session: {e}")
                
                logger.info(f"Participant left: {participant}, reason: {reason}")

            # Add more event handlers for debugging
            @transport.event_handler("on_call_state_updated")
            async def on_call_state_updated(transport, state):
                logger.info(f"Call state updated: {state}")

            logger.info("Starting pipeline runner...")
            runner = PipelineRunner()
            await runner.run(task)

        except Exception as e:
            logger.error(f"Main execution error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())