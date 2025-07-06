"""
RTVI action handlers for the conversational AI pipeline using standardized messaging.
"""

from loguru import logger
from typing import List
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIAction,
    RTVIActionArgument,
)
from services.message_service import get_message_service


async def handle_append_messages(processor, service, arguments):
    """Handle appending messages to the conversation."""
    from services.pipeline_manager import PipelineManager
    
    messages = arguments.get("messages", [])
    logger.info(f"Handling append_messages with {len(messages)} messages")
    
    pipeline_manager = PipelineManager.get_instance()
    context_aggregator = pipeline_manager.get_context_aggregator()
    task = pipeline_manager.get_task()
    
    if messages and context_aggregator and task:
        # Create LLM messages frame and push it
        llm_frame = LLMMessagesAppendFrame(messages=messages)
        await task.queue_frames([llm_frame, context_aggregator.user().get_context_frame()])
        
        logger.debug("Pushed LLM messages frame")
        return True  # Indicate success
    else:
        logger.warning("No messages to append or context aggregator not available")
        return False


async def send_follow_up_questions(questions: List[str], rtvi_processor=None) -> bool:
    """Send follow-up questions to the UI using the message service."""
    message_service = get_message_service()
    
    # Set the RTVI processor if provided
    if rtvi_processor:
        message_service.set_rtvi_processor(rtvi_processor)
    
    # Send follow-up questions using the standardized message protocol
    return await message_service.send_follow_up_questions(questions)


def create_rtvi_actions():
    """Create and return RTVI actions."""
    append_messages_action = RTVIAction(
        service="llm",
        action="append_to_messages",
        arguments=[RTVIActionArgument(name="messages", type="array")],
        result="bool",
        handler=handle_append_messages
    )
    return [append_messages_action]
