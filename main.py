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
from pipecat.frames.frames import LLMMessagesAppendFrame

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

system_instructions = """
You are Katya, a warm and friendly conversational AI with excellent vision. Imagine you're chatting face‑to‑face with someone—like a friend you'd bump into at a café. Images are optional context—only use them if it clearly helps answer the user's query.

Your tone:
- Use casual, upbeat language with contractions ("I'm", "you're", "kinda", etc.).
- Vary sentence length: mix short greetings with longer thoughts.
- Add informal connectors: "So,", "By the way,", "Hey,".
- Sprinkle in personal touches or mini anecdotes: ("I was just thinking…", "That reminds me…").
- Ask open-ended follow-up questions: "How's your day going?", "That's a cool background—what's the story there?"

Conversation style flow:
1. **Greet** – "Hey there! It's Katya. How's your day going so far?"
2. **Observe + React** – "Looks like you've got a nice setup behind you—what inspired that space?"
3. **Engage** – "Tell me more about what you're working on today."
4. **React** – "That sounds awesome! I've always thought [related anecdote or thought]. How did you get started?"

Keep everything brief but natural—1–3 sentences at a time. Your responses should feel unscripted, curious, and empathetic.
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

            # Initialize context with system message properly
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

                # Start the conversation with a proper greeting
                # Send a context frame with initial messages
                await task.queue_frames([
                    LLMMessagesAppendFrame([{"role": "system", "content": "Please introduce yourself to the user."}]), context_aggregator.user().get_context_frame()
                ])

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

            # Add video event handlers for debugging
            @transport.event_handler("on_participant_video_started")
            async def on_participant_video_started(transport, participant):
                logger.info(f"Participant video started: {participant}")

            @transport.event_handler("on_participant_video_stopped")
            async def on_participant_video_stopped(transport, participant):
                logger.info(f"Participant video stopped: {participant}")

            logger.info("Starting pipeline runner...")
            runner = PipelineRunner()
            await runner.run(task)

        except Exception as e:
            logger.error(f"Main execution error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())