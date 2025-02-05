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
from heygen_client import HeyGenClient, NewSessionRequest
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

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        room, token = await configure(session)
        print("ROOM URL", room)
        # Open room URL in default browser
        try:
            webbrowser.open(room)
        except:
            logger.warning("Could not open room URL in browser")

        transport = DailyTransport(
            room,
            token,
            "HeyGen",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=854,
                camera_out_height=480,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            ),
        )

        stt = DeepgramSTTService(
            api_key=settings.DEEPGRAM_API_KEY,
            live_options=LiveOptions(language="en-US"),
        )

        tts = ElevenLabsTTSService(
            api_key=settings.ELEVENLABS_API_KEY, voice_id="nPczCjzI2devNBz1zQrb"
        )

        heygen_client = HeyGenClient(api_key=settings.HEYGEN_API_KEY, session=session)

        session_response = await heygen_client.new_session(
            NewSessionRequest(
                avatarName="Shawn_Therapist_public",
                version="v2",
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

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
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
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
