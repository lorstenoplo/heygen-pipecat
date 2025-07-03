#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.base_transport import BaseTransport
from typing import Any
from config import settings

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def maybe_capture_participant_camera(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant camera video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    if isinstance(transport, DailyTransport):
        await transport.capture_participant_video(
            client["id"], framerate=framerate, video_source="camera"
        )


async def maybe_capture_participant_screen(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant screen video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    if isinstance(transport, DailyTransport):
        await transport.capture_participant_video(
            client["id"], framerate=framerate, video_source="screenVideo"
        )

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
                "Katya",
                DailyParams(
                    audio_out_enabled=True,
                    camera_out_enabled=True,
                    camera_out_width=1280,
                    camera_out_height=1120,
                    vad_enabled=True,
                    transcription_enabled=True,
                    audio_in_sample_rate=16000,
                    audio_out_sample_rate=24000,
                    vad_audio_passthrough=True,
                    video_out_enabled=True,
                ),
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
            
            system_instruction = """You are a Katya in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.
            Always talk in a friendly and engaging manner, as if you were a real person. Use a natural tone and avoid sounding robotic. Your responses should be concise and to the point, while still providing value to the user. Dont mention that you are an AI or a virtual assistant, just act like a real person.
            """

            # Define tools for Gemini service
            search_tool = {"google_search": {}}
            tools = [search_tool]

            # Create Gemini Multimodal Live service with female voice
            gemini_service = GeminiMultimodalLiveLLMService(
                api_key=settings.GOOGLE_API_KEY,
                voice_id="Aoede",
                transcription_enabled=True,
                system_instruction=system_instruction,
                temperature=0.8,
                max_output_tokens=1000,
                tools=tools,
            )

            messages =[{
                "role": "system",
                "content": "Introduce yourself as Katya."
            }]

            from pipecat.services.openai import OpenAILLMContext

            context = OpenAILLMContext(messages) # type: ignore
            context_aggregator = gemini_service.create_context_aggregator(context)

            pipeline = Pipeline(
                [
                    transport.input(),
                    context_aggregator.user(),
                    gemini_service,
                    heygen_video_service,
                    transport.output(),
                    context_aggregator.assistant(),
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

                await maybe_capture_participant_camera(transport, client, framerate=1)
                await maybe_capture_participant_screen(transport, client, framerate=1)

                await task.queue_frames([context_aggregator.user().get_context_frame()])
                await asyncio.sleep(3)
                logger.debug("Unpausing audio and video")
                gemini_service.set_audio_input_paused(False)
                gemini_service.set_video_input_paused(False)

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.info(f"Client disconnected")
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