#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat HeyGen Video Bot Example.

The example runs a voice AI bot with HeyGen video avatars that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- ElevenLabs (Text-to-Speech)
- HeyGen (Video Avatar)

The example connects between client and server using transport services.

Run the bot using::

    python main.py
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="21m00Tcm4TlvDq8ikWAM",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        heyGen = HeyGenVideoService(api_key=os.getenv("HEYGEN_API_KEY"), session=session)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your output will be converted to audio so don't include special characters in your answers. Be succinct and respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                heyGen,  # Avatar
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Start by saying 'Hello' and then a short greeting.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import argparse
    import sys
    import asyncio
    
    # Check if we have daily-specific arguments (-u for room URL, -t for token)
    if '-u' in sys.argv and '-t' in sys.argv:
        # Extract room URL and token from command line
        url_index = sys.argv.index('-u') + 1
        token_index = sys.argv.index('-t') + 1
        room_url = sys.argv[url_index]
        token = sys.argv[token_index]
        
        # Set environment variables for direct Daily connection
        os.environ['DAILY_SAMPLE_ROOM_URL'] = room_url
        os.environ['DAILY_SAMPLE_ROOM_TOKEN'] = token
        
        # Remove these arguments and use direct mode
        sys.argv = [arg for i, arg in enumerate(sys.argv) 
                   if arg not in ['-u', '-t'] and i not in [url_index, token_index]]
        sys.argv.extend(['-d'])  # Add --direct flag for Daily direct connection
    
    # Intercept command line arguments and force daily transport
    # Remove any transport arguments that aren't daily and set to daily
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['-t', '--transport']:
            # Skip the transport argument and its value, we'll add daily later
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                i += 1  # Skip the value too
        elif arg.startswith('--transport='):
            # Skip --transport=value format
            pass
        else:
            filtered_args.append(arg)
        i += 1
    
    # Update sys.argv with filtered arguments
    sys.argv = filtered_args
    
    # Add daily transport as default if no direct mode
    if '-d' not in sys.argv and '--direct' not in sys.argv:
        sys.argv.extend(['-t', 'daily'])
    
    from pipecat.runner.run import main
    main()
