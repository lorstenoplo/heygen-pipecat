"""
RTVI action handlers for the conversational AI pipeline.
"""

from loguru import logger
from datetime import datetime
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIAction,
    RTVIActionArgument,
    RTVIServerMessageFrame,
)


async def handle_append_messages(processor, service, arguments):
    """Handle appending messages to the conversation."""
    from ..services.pipeline_manager import PipelineManager
    
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


async def send_follow_up_questions(questions: list, rtvi_processor):
    """Send follow-up questions to the UI - DELAYED to avoid timing issues."""
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
