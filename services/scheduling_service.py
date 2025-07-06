"""
Scheduling service with email and calendar integration using standardized messaging protocol.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import resend
try:
    from icalendar import Calendar, Event as iCalEvent
    CALENDAR_AVAILABLE = True
except ImportError:
    Calendar = None
    iCalEvent = None
    CALENDAR_AVAILABLE = False
from loguru import logger

from messaging.protocol import (
    StandardMessage, MessageType, Priority,
    create_tool_call_start, create_tool_call_progress, create_tool_call_result, create_tool_call_error,
    create_process_start, create_process_step, create_process_complete, create_process_error,
    create_ui_update, create_agent_state
)
from services.message_service import get_message_service
from config.settings import settings


@dataclass
class SchedulingRequest:
    """Data structure for scheduling requests"""
    email: str
    date: str  # YYYY-MM-DD format
    time: str  # HH:MM format
    service_type: str
    duration_minutes: int = 60
    timezone: str = "UTC"
    notes: Optional[str] = None
    
    def to_datetime(self) -> datetime:
        """Convert date and time to datetime object"""
        return datetime.fromisoformat(f"{self.date}T{self.time}:00")


@dataclass
class SchedulingResult:
    """Result of scheduling operation"""
    success: bool
    booking_id: Optional[str] = None
    confirmation_email_sent: bool = False
    calendar_invite_sent: bool = False
    error_message: Optional[str] = None


class SchedulingService:
    """Service for handling appointment scheduling with email and calendar integration"""
    
    def __init__(self):
        self.resend_client = None
        if settings.RESEND_API_KEY:
            resend.api_key = settings.RESEND_API_KEY
            self.resend_client = resend
        self.message_service = get_message_service()
        
    async def schedule_appointment(self, request: SchedulingRequest) -> SchedulingResult:
        """
        Main scheduling method that follows the protocol.py message flow
        """
        booking_id = f"booking_{uuid.uuid4().hex[:8]}"
        process_id = f"schedule_proc_{uuid.uuid4().hex[:8]}"
        
        try:
            # Send process start message
            await self._send_message(create_process_start(
                process_id=process_id,
                name="appointment_scheduling",
                total_steps=4,
                description=f"Scheduling {request.service_type} appointment for {request.email}"
            ))
            
            # Step 1: Validate request
            await self._send_step_message(process_id, "validate", "Validating Request", 0, 4, "running")
            validation_result = await self._validate_request(request)
            
            if not validation_result["valid"]:
                await self._send_error_message(process_id, validation_result["error"])
                return SchedulingResult(success=False, error_message=validation_result["error"])
            
            await self._send_step_message(process_id, "validate", "Request Validated", 0, 4, "completed")
            
            # Step 2: Check availability (mock for now)
            await self._send_step_message(process_id, "availability", "Checking Availability", 1, 4, "running")
            availability_result = await self._check_availability(request)
            
            if not availability_result["available"]:
                await self._send_error_message(process_id, "Requested time slot is not available")
                return SchedulingResult(success=False, error_message="Time slot not available")
            
            await self._send_step_message(process_id, "availability", "Time Slot Confirmed", 1, 4, "completed")
            
            # Step 3: Create calendar invite
            await self._send_step_message(process_id, "calendar", "Creating Calendar Invite", 2, 4, "running")
            calendar_invite = self._create_calendar_invite(request, booking_id)
            await self._send_step_message(process_id, "calendar", "Calendar Invite Created", 2, 4, "completed")
            
            # Step 4: Send confirmation email
            await self._send_step_message(process_id, "email", "Sending Confirmation Email", 3, 4, "running")
            email_result = await self._send_confirmation_email(request, booking_id, calendar_invite)
            
            if email_result["success"]:
                await self._send_step_message(process_id, "email", "Confirmation Email Sent", 3, 4, "completed")
            else:
                await self._send_step_message(process_id, "email", "Email Failed", 3, 4, "failed")
                logger.warning(f"Email sending failed: {email_result.get('error')}")
            
            # Complete process
            result = SchedulingResult(
                success=True,
                booking_id=booking_id,
                confirmation_email_sent=email_result["success"],
                calendar_invite_sent=email_result["success"]
            )
            
            await self._send_message(create_process_complete(
                process_id=process_id,
                name="appointment_scheduling",
                result={
                    "booking_id": booking_id,
                    "email_sent": email_result["success"],
                    "calendar_invite_included": True,
                    "appointment_time": f"{request.date} {request.time}",
                    "service": request.service_type
                }
            ))
            
            # Send UI update for booking confirmation
            await self._send_message(create_ui_update(
                component_id="booking_confirmation",
                action="show_confirmation",
                data={
                    "booking_id": booking_id,
                    "status": "confirmed",
                    "email": request.email,
                    "datetime": f"{request.date} {request.time}",
                    "service": request.service_type
                }
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"Scheduling error: {e}")
            await self._send_error_message(process_id, str(e))
            return SchedulingResult(success=False, error_message=str(e))
    
    async def _validate_request(self, request: SchedulingRequest) -> Dict[str, Any]:
        """Validate the scheduling request"""
        try:
            # Check email format
            if "@" not in request.email:
                return {"valid": False, "error": "Invalid email format"}
            
            # Check date/time format and ensure it's in the future
            appointment_time = request.to_datetime()
            if appointment_time <= datetime.now():
                return {"valid": False, "error": "Appointment time must be in the future"}
            
            # Check service type
            valid_services = ["consultation", "demo", "onboarding", "support", "training"]
            if request.service_type.lower() not in valid_services:
                return {"valid": False, "error": f"Invalid service type. Must be one of: {', '.join(valid_services)}"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _check_availability(self, request: SchedulingRequest) -> Dict[str, Any]:
        """Check if the requested time slot is available (mock implementation)"""
        # In a real implementation, this would check against a calendar system
        appointment_time = request.to_datetime()
        
        # Mock business hours check (9 AM to 5 PM)
        if appointment_time.hour < 9 or appointment_time.hour >= 17:
            return {"available": False, "reason": "Outside business hours (9 AM - 5 PM)"}
        
        # Mock weekend check
        if appointment_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return {"available": False, "reason": "Weekends not available"}
        
        return {"available": True}
    
    def _create_calendar_invite(self, request: SchedulingRequest, booking_id: str) -> bytes:
        """Create an iCalendar invite"""
        if not CALENDAR_AVAILABLE or Calendar is None or iCalEvent is None:
            logger.warning("Calendar library not available, returning empty bytes")
            return b""
            
        try:
            from icalendar import vCalAddress, vText
            import zoneinfo
            
            cal = Calendar()
            cal.add('prodid', '-//Kreyn AI Scheduling//Appointment//EN')
            cal.add('version', '2.0')
            cal.add('calscale', 'GREGORIAN')
            
            event = iCalEvent()
            event.add('uid', booking_id)
            event.add('summary', f'{request.service_type.title()} Appointment')
            event.add('description', f"""Appointment Details:
- Service: {request.service_type.title()}
- Duration: {request.duration_minutes} minutes
- Booking ID: {booking_id}

{request.notes or 'No additional notes'}

If you need to reschedule or cancel, please contact us at admin@kreyn.ai.
            """.strip())
            
            # Handle timezone properly
            try:
                tz = zoneinfo.ZoneInfo(request.timezone)
            except Exception:
                tz = zoneinfo.ZoneInfo("UTC")
                logger.warning(f"Invalid timezone {request.timezone}, using UTC")
            
            start_time = request.to_datetime().replace(tzinfo=tz)
            end_time = start_time + timedelta(minutes=request.duration_minutes)
            
            event.add('dtstart', start_time)
            event.add('dtend', end_time)
            event.add('dtstamp', datetime.now(tz=zoneinfo.ZoneInfo("UTC")))
            event.add('status', 'CONFIRMED')
            
            # Add organizer with proper format
            organizer = vCalAddress('MAILTO:admin@kreyn.ai')
            organizer.params['cn'] = vText('Kreyn AI')
            organizer.params['role'] = vText('CHAIR')
            event['organizer'] = organizer
            
            # Add attendee with proper format
            attendee = vCalAddress(f'MAILTO:{request.email}')
            attendee.params['cn'] = vText(request.email.split('@')[0])
            attendee.params['ROLE'] = vText('REQ-PARTICIPANT')
            attendee.params['RSVP'] = vText('TRUE')
            event['attendee'] = attendee
            
            cal.add_component(event)
            return cal.to_ical()
        except Exception as e:
            logger.error(f"Failed to create calendar invite: {e}")
            return b""
    
    async def _send_confirmation_email(self, request: SchedulingRequest, booking_id: str, calendar_invite: bytes) -> Dict[str, Any]:
        """Send confirmation email with calendar invite"""
        if not self.resend_client:
            logger.warning("Resend API key not configured, skipping email")
            return {"success": False, "error": "Email service not configured"}
        
        try:
            # Create beautiful HTML email
            html_content = self._create_email_html(request, booking_id)
            
            # Prepare email parameters according to resend API
            email_params: resend.Emails.SendParams = {
                "from": "Kreyn AI <admin@kreyn.ai>",
                "to": [request.email],
                "subject": f"✅ Your {request.service_type.title()} Appointment is Confirmed",
                "html": html_content
            }
            
            # Add calendar attachment if available
            if calendar_invite:
                import base64
                email_params["attachments"] = [
                    {
                        "filename": "appointment.ics",
                        "content": base64.b64encode(calendar_invite).decode(),
                        "content_type": "text/calendar"
                    }
                ]
            
            # Send email using resend
            try:
                email_response = resend.Emails.send(email_params)
                logger.info(f"Confirmation email sent to {request.email}, ID: {email_response.get('id')}")
                return {"success": True, "email_id": email_response.get('id')}
            except Exception as resend_error:
                logger.error(f"Failed to send email via Resend: {resend_error}")
                return {"success": False, "error": str(resend_error)}
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_email_html(self, request: SchedulingRequest, booking_id: str) -> str:
        """Create beautiful HTML email content"""
        appointment_time = request.to_datetime()
        formatted_date = appointment_time.strftime("%A, %B %d, %Y")
        formatted_time = appointment_time.strftime("%I:%M %p")
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Appointment Confirmation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 12px 12px 0 0; }}
        .content {{ background: white; padding: 30px; border: 1px solid #e1e5e9; border-top: none; }}
        .appointment-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .detail-row {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 8px 0; border-bottom: 1px solid #e9ecef; }}
        .detail-label {{ font-weight: bold; color: #495057; }}
        .detail-value {{ color: #6c757d; }}
        .cta-button {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 20px 0; }}
        .footer {{ text-align: center; color: #6c757d; font-size: 14px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; }}
        .success-icon {{ font-size: 48px; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="success-icon">✅</div>
        <h1>Appointment Confirmed!</h1>
        <p>Your {request.service_type.title()} appointment has been successfully scheduled</p>
    </div>
    
    <div class="content">
        <div class="appointment-card">
            <h3 style="margin-top: 0; color: #495057;">Appointment Details</h3>
            
            <div class="detail-row">
                <span class="detail-label">Service:</span>
                <span class="detail-value">{request.service_type.title()}</span>
            </div>
            
            <div class="detail-row">
                <span class="detail-label">Date:</span>
                <span class="detail-value">{formatted_date}</span>
            </div>
            
            <div class="detail-row">
                <span class="detail-label">Time:</span>
                <span class="detail-value">{formatted_time} ({request.timezone})</span>
            </div>
            
            <div class="detail-row">
                <span class="detail-label">Duration:</span>
                <span class="detail-value">{request.duration_minutes} minutes</span>
            </div>
            
            <div class="detail-row">
                <span class="detail-label">Booking ID:</span>
                <span class="detail-value">{booking_id}</span>
            </div>
            
            {f'<div class="detail-row"><span class="detail-label">Notes:</span><span class="detail-value">{request.notes}</span></div>' if request.notes else ''}
        </div>
        
        <h3>What's Next?</h3>
        <ul>
            <li>📅 <strong>Add to Calendar:</strong> Click the attached calendar file (.ics) to add this appointment to your calendar</li>
            <li>🔗 <strong>Join Link:</strong> We'll send you the meeting link 15 minutes before the appointment</li>
            <li>📞 <strong>Need Changes?:</strong> Reply to this email or contact us at admin@kreyn.ai to reschedule</li>
        </ul>
        
        <div style="text-align: center;">
            <a href="mailto:admin@kreyn.ai" class="cta-button">Contact Support</a>
        </div>
        
        <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 6px; padding: 15px; margin: 20px 0;">
            <strong>💡 Tip:</strong> Save this email for your records and join the meeting 5 minutes early for the best experience.
        </div>
    </div>
    
    <div class="footer">
        <p>Thank you for choosing Kreyn AI!</p>
        <p>© 2025 Kreyn AI. All rights reserved.</p>
        <p>For support, contact us at <a href="mailto:admin@kreyn.ai">admin@kreyn.ai</a></p>
    </div>
</body>
</html>
        """
    
    async def _send_message(self, message: StandardMessage):
        """Send a message using the message service"""
        await self.message_service.send_message(message)
    
    async def _send_step_message(self, process_id: str, step_id: str, step_name: str, 
                                step_index: int, total_steps: int, status: str):
        """Send a process step message"""
        from messaging.protocol import create_process_step
        
        message = create_process_step(
            process_id=process_id,
            step_id=step_id,
            step_name=step_name,
            step_index=step_index,
            total_steps=total_steps,
            status=status
        )
        await self._send_message(message)
    
    async def _send_error_message(self, process_id: str, error: str):
        """Send a process error message"""
        message = create_process_error(process_id=process_id, error=error)
        await self._send_message(message)


# Global instance
_scheduling_service = None

def get_scheduling_service() -> SchedulingService:
    """Get the global scheduling service instance"""
    global _scheduling_service
    if _scheduling_service is None:
        _scheduling_service = SchedulingService()
    return _scheduling_service
