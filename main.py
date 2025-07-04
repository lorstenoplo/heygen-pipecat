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
from pipecat.services.openai_realtime_beta.openai import (
    OpenAIRealtimeBetaLLMService,
)
from pipecat.services.openai_realtime_beta.events import SessionProperties
from pipecat.transports.services.daily import DailyParams, DailyTransport
from config import settings
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
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
                    vad_audio_passthrough=True,
                    video_out_enabled=True,
                    audio_in_enabled=True,
                ),
            )

            # Create HeyGen client
            heygen_client = HeyGenClient(api_key=settings.HEYGEN_API_KEY, session=session)
            
            session_response = await heygen_client.new_session(
                NewSessionRequest(
                    avatarName="Katya_Chair_Sitting_public",
                    version="v1",
                    quality=AvatarQuality.high,
                )
            )
            
            await heygen_client.start_session(session_response.session_id)

            heygen_video_service = HeyGenVideoService(
                session_id=session_response.session_id,
                session_token=session_response.access_token,
                session=session,
                realtime_endpoint=session_response.realtime_endpoint,
                livekit_room_url=session_response.url,
            )

            # OpenAI Realtime LLM for ultra-fast responses
            session_properties = SessionProperties(
                input_audio_transcription=InputAudioTranscription(),
                turn_detection=SemanticTurnDetection(),
                input_audio_noise_reduction=InputAudioNoiseReduction(type="near_field"),
                instructions="""You are Katya, a friendly and engaging person who loves meeting new people. You have a warm, conversational personality and enjoy getting to know others through natural dialogue.

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

When someone asks about visual things or what you can see, politely explain that you can't see right now but you'd love to hear them describe what's happening or what they're doing.

Start conversations by introducing yourself as Katya and asking how their day is going. Be genuinely interested in getting to know them."""
            )

            llm = OpenAIRealtimeBetaLLMService(
                api_key=settings.OPENAI_API_KEY,
                session_properties=session_properties,
                start_audio_paused=False,
            )

            # Simple context for introduction
            context = OpenAILLMContext(
                [{"role": "user", "content": "Introduce yourself as Katya and ask how their day is going"}]
            )
            context_aggregator = llm.create_context_aggregator(context)

            # Streamlined pipeline for maximum speed
            pipeline = Pipeline(
                [
                    transport.input(),
                    context_aggregator.user(),
                    llm,
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
                
                # Start conversation immediately
                await task.queue_frames([context_aggregator.user().get_context_frame()])

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
                try:
                    await heygen_client.stop_session(session_response.session_id)
                    logger.info(f"HeyGen session stopped for participant: {participant['id']}")
                except Exception as e:
                    logger.error(f"Failed to stop HeyGen session: {e}")
                
                logger.info(f"Participant left: {participant}, reason: {reason}")

            logger.info("Starting pipeline runner...")
            runner = PipelineRunner()
            await runner.run(task)

        except Exception as e:
            logger.error(f"Main execution error: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())