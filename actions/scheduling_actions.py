"""
Simple RTVI action for handling scheduling data from frontend.
"""

from loguru import logger
from typing import Dict, Any

from services.scheduling_service import get_scheduling_service, SchedulingRequest


async def handle_schedule_appointment(processor, service, arguments):
    """
    Handle scheduling appointment with data from frontend.
    
    Expected arguments:
    - email: User's email address
    - date: Date in YYYY-MM-DD format
    - time: Time in HH:MM format
    - service_type: Type of service being scheduled
    - duration_minutes: Duration in minutes (optional, defaults to 60)
    - timezone: User's timezone (optional, defaults to UTC)
    - notes: Additional notes (optional)
    """
    try:
        # Extract scheduling data from arguments
        email = arguments.get("email")
        date = arguments.get("date")
        time = arguments.get("time")
        service_type = arguments.get("service_type", "consultation")
        duration_minutes = arguments.get("duration_minutes", 60)
        timezone = arguments.get("timezone", "UTC")
        notes = arguments.get("notes", "")
        
        logger.info(f"Processing scheduling request: {service_type} for {email} on {date} at {time}")
        
        # Validate required fields
        if not all([email, date, time]):
            logger.error("Missing required scheduling fields")
            return {
                "success": False,
                "error": "Missing required fields: email, date, or time"
            }
        
        # Create scheduling request
        scheduling_request = SchedulingRequest(
            email=email,
            date=date,
            time=time,
            service_type=service_type,
            duration_minutes=int(duration_minutes),
            timezone=timezone,
            notes=notes
        )
        
        # Process the scheduling
        scheduling_service = get_scheduling_service()
        result = await scheduling_service.schedule_appointment(scheduling_request)
        
        # Return simple result
        return {
            "success": result.success,
            "booking_id": result.booking_id,
            "email_sent": result.confirmation_email_sent,
            "calendar_invite_sent": result.calendar_invite_sent,
            "error": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error in handle_schedule_appointment: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def create_scheduling_rtvi_actions():
    """Create RTVI actions for scheduling functionality"""
    from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIActionArgument
    
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
    
    return [schedule_appointment_action]
