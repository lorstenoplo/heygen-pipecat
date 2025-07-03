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

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
                ),
            )

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

            llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
                },
            ]

            context = OpenAILLMContext(messages) # type: ignore
            context_aggregator = llm.create_context_aggregator(context)

            pipeline = Pipeline(
                [
                    transport.input(),
                    stt,
                    context_aggregator.user(),
                    llm,
                    tts,
                    heygen_video_service,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            )

            task = PipelineTask(
                pipeline,
                params = PipelineParams(
                    allow_interruptions=True,
                    enable_usage_metrics=True,
                ),
            )

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"First participant joined: {participant}")
                await transport.capture_participant_transcription(participant["id"])
                await task.queue_frames([context_aggregator.user().get_context_frame()])

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