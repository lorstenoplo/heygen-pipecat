import asyncio
import os
import sys

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
from pipecat.frames.frames import LLMMessagesAppendFrame, StartFrame, EndFrame
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIProcessor,
    RTVIServerMessageFrame,
    RTVIAction,
    RTVIActionArgument,
    RTVIObserver,
)

from pydantic import BaseModel
from typing import List
import openai
from datetime import datetime

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Global references
rtvi_processor = None
transport_instance = None
task = None
context_aggregator = None
follow_up_llm = None
# context for follow-up questions
context = []

class FollowUpQuestions(BaseModel):
    questions: List[str]

class OpenAIFollowUpProcessor:
    """Follow-up question generator using OpenAI API with Pydantic."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    async def generate_follow_ups(self, assistant_response: str):
        """Generate follow-up questions using OpenAI with structured output."""
        global context

        context.append(assistant_response)
        if len(context) > 5:
            context.pop(0)

        try:
            prompt = f"""Based on this AI assistant response, generate 2-3 short, relevant follow-up questions that users might naturally ask next.

            Assistant Response: "{assistant_response}"
            Context: {context}

            Generate natural, conversational questions that would logically follow from this response."""

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=FollowUpQuestions,
                temperature=0.7
            )

            questions_obj = response.choices[0].message.parsed
            if questions_obj and questions_obj.questions and len(questions_obj.questions) >= 2:
                return questions_obj.questions[:3]
            
            # Fallback
            return ["Tell me more about that", "What else should I know?"]

        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return ["Can you explain further?", "What's next?"]


async def handle_append_messages(processor, service, arguments):
    """Handle appending messages to the conversation."""
    global context_aggregator
    
    messages = arguments.get("messages", [])
    logger.info(f"Handling append_messages with {len(messages)} messages")
    
    if messages and context_aggregator and task:
        # Create LLM messages frame and push it
        llm_frame = LLMMessagesAppendFrame(messages=messages)
        await task.queue_frames([llm_frame, context_aggregator.user().get_context_frame()])
        
        logger.debug("Pushed LLM messages frame")
        return True  # Indicate success
    else:
        logger.warning("No messages to append or context aggregator not available")
        return False


async def send_follow_up_questions(questions: list):
    """Send follow-up questions to the UI - DELAYED to avoid timing issues."""
    global rtvi_processor
    
    if rtvi_processor and questions:
        try:            
            frame = RTVIServerMessageFrame(
                data={
                    "type": "ui_update_follow_up",
                    "payload": {
                        "questions": questions,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            )
            await rtvi_processor.push_frame(frame)
            logger.debug(f"Sent follow-up questions: {questions}")
        except Exception as e:
            logger.error(f"Error sending follow-up questions: {e}")

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
2. **Engage** – "Tell me more about what you're working on today."
3. **React** – "That sounds awesome! I've always thought [related anecdote or thought]. How did you get started?"

Keep everything brief but natural—1–3 sentences at a time. Your responses should feel unscripted, curious, and empathetic.
"""

async def main():
    global rtvi_processor, transport_instance, task, context_aggregator, follow_up_llm

    async with aiohttp.ClientSession() as session:
        try:
            room, token = await configure(session)
            logger.info(f"Room configured: {room}")

            # Transport setup
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
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.0)),
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
            except Exception as e:
                logger.error(f"Failed to create HeyGen session: {e}")
                raise

            # Start session
            try:
                await heygen_client.start_session(session_response.session_id)
                logger.info(f"HeyGen session started: {session_response.session_id}")
            except Exception as e:
                logger.error(f"Failed to start HeyGen session: {e}")
                raise

            # Create video service
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

            # Follow-up processor
            follow_up_processor = OpenAIFollowUpProcessor()

            # RTVI processor setup - FIXED: Include in main pipeline
            rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
            rtvi_processor = rtvi

            # Register RTVI actions
            append_messages_action = RTVIAction(
                service="llm",
                action="append_to_messages",
                arguments=[RTVIActionArgument(name="messages", type="array")],
                result="bool",
                handler=handle_append_messages
            )
            rtvi_processor.register_action(append_messages_action)

            # Transcript processor
            from pipecat.processors.transcript_processor import TranscriptProcessor
            transcript = TranscriptProcessor()

            # FIXED: Include RTVI in the main pipeline flow to ensure proper StartFrame handling
            pipeline = Pipeline(
                [
                    transport.input(),           # Input from transport
                    rtvi,                       # RTVI processor - MOVED TO MAIN PIPELINE
                    stt,                        # Speech to text
                    transcript.user(),          # User transcript
                    context_aggregator.user(),  # Add user message + vision to context
                    llm,                       # LLM processes text + vision
                    tts,                       # Text to speech
                    transcript.assistant(),     # Assistant transcript
                    context_aggregator.assistant(), # Add assistant response to context
                    heygen_video_service,      # Video output
                    transport.output(),        # Output to transport
                ]
            )

            # FIXED: Remove RTVIObserver since RTVI is now in the main pipeline
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_usage_metrics=True,
                ),
                observers=[RTVIObserver(rtvi)],
            )

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, client):
                logger.info(f"Client connected: {client}")

                # FIXED: Wait for RTVI to be ready before sending initial message
                await asyncio.sleep(0.1)  # Small delay to ensure RTVI is started
                
                if task and context_aggregator:
                    await task.queue_frames([
                        LLMMessagesAppendFrame([{"role": "user", "content": "Hello"}]),
                        context_aggregator.user().get_context_frame()
                    ])

            @transcript.event_handler("on_transcript_update")
            async def handle_transcript_update(processor, frame):
                """Handle transcript updates and generate follow-ups for assistant responses."""
                for message in frame.messages:
                    logger.info(f"TRANSCRIPT: {message.role}: {message.content}")
                    
                    # Generate follow-ups for assistant responses
                    if message.role == "assistant" and message.content and len(message.content.strip()) > 10:
                        asyncio.create_task(
                            delayed_follow_up_generation(message.content)
                        )

            async def delayed_follow_up_generation(assistant_response: str):
                """Generate follow-ups with proper delay to avoid timing issues."""
                try:
                    questions = await follow_up_processor.generate_follow_ups(assistant_response)
                    await send_follow_up_questions(questions)
                except Exception as e:
                    logger.error(f"Error in delayed follow-up generation: {e}")

            @rtvi.event_handler("on_client_ready")
            async def on_client_ready(rtvi):
                logger.info("RTVI client ready")
                await rtvi.set_bot_ready()

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.info(f"Client disconnected: {client}")
                if task:
                    try:
                        await task.cancel()
                        logger.info("Pipeline task cancelled successfully")
                    except Exception as e:
                        logger.error(f"Error cancelling pipeline task: {e}")

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

            @transport.event_handler("on_call_state_updated")
            async def on_call_state_updated(transport, state):
                logger.info(f"Call state updated: {state}")

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