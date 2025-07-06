"""
Event handlers for transport and pipeline events.
"""

import asyncio
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from services.follow_up_service import OpenAIFollowUpProcessor
from actions.rtvi_actions import send_follow_up_questions
from services.pipeline_manager import PipelineManager


class EventHandlers:
    """Container for all event handlers."""
    
    def __init__(self, follow_up_processor: OpenAIFollowUpProcessor):
        self.follow_up_processor = follow_up_processor
        self.pipeline_manager = PipelineManager.get_instance()
    
    async def on_client_connected(self, transport, client):
        """Handle client connection."""
        logger.info(f"Client connected: {client}")

        # Wait for RTVI to be ready before sending initial message
        await asyncio.sleep(0.1)
        
        task = self.pipeline_manager.get_task()
        context_aggregator = self.pipeline_manager.get_context_aggregator()
        
        if task and context_aggregator:
            await task.queue_frames([
                LLMMessagesAppendFrame([{"role": "user", "content": "Hello"}]),
                context_aggregator.user().get_context_frame()
            ])

    async def on_client_disconnected(self, transport, client):
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {client}")
        task = self.pipeline_manager.get_task()
        if task:
            try:
                await task.cancel()
                logger.info("Pipeline task cancelled successfully")
            except Exception as e:
                logger.error(f"Error cancelling pipeline task: {e}")

    async def on_first_participant_joined(self, transport, participant):
        """Handle first participant joining."""
        logger.info(f"First participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    async def on_participant_left(self, transport, participant, reason, heygen_client, session_id):
        """Handle participant leaving."""
        # Stop the heygen session when the participant leaves
        try:
            await heygen_client.stop_session(session_id)
            logger.info(f"HeyGen session stopped for participant: {participant['id']}")
        except Exception as e:
            logger.error(f"Failed to stop HeyGen session: {e}")
        
        logger.info(f"Participant left: {participant}, reason: {reason}")

    async def on_call_state_updated(self, transport, state):
        """Handle call state updates."""
        logger.info(f"Call state updated: {state}")

    async def on_participant_video_started(self, transport, participant):
        """Handle participant video start."""
        logger.info(f"Participant video started: {participant}")

    async def on_participant_video_stopped(self, transport, participant):
        """Handle participant video stop."""
        logger.info(f"Participant video stopped: {participant}")

    async def handle_transcript_update(self, processor, frame):
        """Handle transcript updates and generate follow-ups for assistant responses."""
        for message in frame.messages:
            logger.info(f"TRANSCRIPT: {message.role}: {message.content}")
            
            # Generate follow-ups for assistant responses
            if message.role == "assistant" and message.content and len(message.content.strip()) > 10:
                asyncio.create_task(
                    self._delayed_follow_up_generation(message.content)
                )

    async def _delayed_follow_up_generation(self, assistant_response: str):
        """Generate follow-ups with proper delay to avoid timing issues."""
        try:
            questions = await self.follow_up_processor.generate_follow_ups(assistant_response)
            rtvi_processor = self.pipeline_manager.get_rtvi_processor()
            await send_follow_up_questions(questions, rtvi_processor)
        except Exception as e:
            logger.error(f"Error in delayed follow-up generation: {e}")

    async def on_client_ready(self, rtvi):
        """Handle RTVI client ready."""
        logger.info("RTVI client ready")
        await rtvi.set_bot_ready()
