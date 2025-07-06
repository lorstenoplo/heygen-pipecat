"""
Main application entry point for the HeyGen-Pipecat conversational AI.
"""

import asyncio
import sys
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner

from runner import configure
from services.service_factory import ServiceFactory
from services.follow_up_service import OpenAIFollowUpProcessor
from handlers.event_handlers import EventHandlers
from config.pipeline_config import PipelineConfig

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    """Main application entry point."""
    async with aiohttp.ClientSession() as session:
        try:
            # Configure room and token
            room, token = await configure(session)
            logger.info(f"Room configured: {room}")

            # Create services
            transport = ServiceFactory.create_transport(room, token)
            stt = ServiceFactory.create_stt_service()
            tts = ServiceFactory.create_tts_service()
            llm = ServiceFactory.create_llm_service()
            
            # Create HeyGen services
            heygen_client = await ServiceFactory.create_heygen_client(session)
            session_response = await ServiceFactory.create_heygen_session(heygen_client)
            heygen_video_service = await ServiceFactory.create_heygen_video_service(session_response, session)

            # Create context and processors
            context = ServiceFactory.create_llm_context()
            context_aggregator = llm.create_context_aggregator(context)
            
            rtvi = ServiceFactory.create_rtvi_processor()
            transcript = ServiceFactory.create_transcript_processor()
            
            # Create follow-up processor and event handlers
            follow_up_processor = OpenAIFollowUpProcessor()
            event_handlers = EventHandlers(follow_up_processor)

            # Setup pipeline
            pipeline = PipelineConfig.create_pipeline(
                transport, rtvi, stt, transcript, context_aggregator,
                llm, tts, heygen_video_service
            )
            
            task = PipelineConfig.create_task(pipeline, rtvi)
            
            # Setup RTVI actions and pipeline manager
            PipelineConfig.setup_rtvi_actions(rtvi)
            PipelineConfig.setup_pipeline_manager(rtvi, task, context_aggregator)

            # Register event handlers using the correct method names
            transport.add_event_handler("on_client_connected", event_handlers.on_client_connected)
            transport.add_event_handler("on_client_disconnected", event_handlers.on_client_disconnected)
            transport.add_event_handler("on_first_participant_joined", event_handlers.on_first_participant_joined)
            transport.add_event_handler("on_participant_left", 
                lambda transport, participant, reason: event_handlers.on_participant_left(
                    transport, participant, reason, heygen_client, session_response.session_id
                ))
            transport.add_event_handler("on_call_state_updated", event_handlers.on_call_state_updated)
            transport.add_event_handler("on_participant_video_started", event_handlers.on_participant_video_started)
            transport.add_event_handler("on_participant_video_stopped", event_handlers.on_participant_video_stopped)
            
            # Use the event_handler decorator for transcript
            @transcript.event_handler("on_transcript_update")
            async def handle_transcript_update(processor, frame):
                await event_handlers.handle_transcript_update(processor, frame)
            
            @rtvi.event_handler("on_client_ready")
            async def on_client_ready(rtvi_instance):
                await event_handlers.on_client_ready(rtvi_instance)

            logger.info("Starting pipeline runner...")
            runner = PipelineRunner()
            await runner.run(task)

        except Exception as e:
            logger.error(f"Main execution error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
