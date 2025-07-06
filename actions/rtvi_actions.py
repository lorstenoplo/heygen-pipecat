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


async def handle_schedule_appointment(processor, service, arguments):
    """Handle appointment scheduling from frontend."""
    from services.scheduling_service import get_scheduling_service, SchedulingRequest
    from loguru import logger
    
    logger.info(f"Handling schedule_appointment with arguments: {arguments}")
    
    try:
        # Extract scheduling data from arguments
        email = arguments.get("email", "")
        date = arguments.get("date", "")
        time = arguments.get("time", "")
        service_type = arguments.get("service_type", "consultation")
        duration_minutes = arguments.get("duration_minutes", 60)
        timezone = arguments.get("timezone", "UTC")
        notes = arguments.get("notes", "")
        
        # Validate required fields
        if not email or not date or not time:
            logger.error("Missing required fields for scheduling")
            return {"success": False, "error": "Email, date, and time are required"}
        
        # Create scheduling request
        request = SchedulingRequest(
            email=email,
            date=date,
            time=time,
            service_type=service_type,
            duration_minutes=duration_minutes,
            timezone=timezone,
            notes=notes if notes else None
        )
        
        # Get scheduling service and process the appointment
        scheduling_service = get_scheduling_service()
        result = await scheduling_service.schedule_appointment(request)
        
        if result.success:
            logger.info(f"Appointment scheduled successfully: {result.booking_id}")
            return {
                "success": True,
                "booking_id": result.booking_id,
                "email_sent": result.confirmation_email_sent,
                "calendar_invite_sent": result.calendar_invite_sent,
                "message": f"Appointment scheduled for {date} at {time}"
            }
        else:
            logger.error(f"Scheduling failed: {result.error_message}")
            return {
                "success": False,
                "error": result.error_message
            }
            
    except Exception as e:
        logger.error(f"Error in handle_schedule_appointment: {e}")
        return {"success": False, "error": str(e)}


def create_rtvi_actions():
    """Create and return RTVI actions."""
    append_messages_action = RTVIAction(
        service="llm",
        action="append_to_messages",
        arguments=[RTVIActionArgument(name="messages", type="array")],
        result="bool",
        handler=handle_append_messages
    )
    
    schedule_appointment_action = RTVIAction(
        service="scheduling",
        action="schedule_appointment",
        arguments=[
            RTVIActionArgument(name="email", type="string"),
            RTVIActionArgument(name="date", type="string"),
            RTVIActionArgument(name="time", type="string"),
            RTVIActionArgument(name="service_type", type="string"),
            RTVIActionArgument(name="duration_minutes", type="number"),
            RTVIActionArgument(name="timezone", type="string"),
            RTVIActionArgument(name="notes", type="string")
        ],
        result="object",
        handler=handle_schedule_appointment
    )
    
    return [append_messages_action, schedule_appointment_action]
