"""
Simple scheduling service with email and calendar integration.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
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
    """Simple service for handling appointment scheduling with email and calendar integration"""
    
    def __init__(self):
        self.resend_client = None
        if settings.RESEND_API_KEY:
            resend.api_key = settings.RESEND_API_KEY
            self.resend_client = resend
        
    async def schedule_appointment(self, request: SchedulingRequest) -> SchedulingResult:
        """
        Simple scheduling method that validates, creates calendar invite, and sends email
        """
        booking_id = f"booking_{uuid.uuid4().hex[:8]}"
        
        try:
            # Validate request
            validation_result = await self._validate_request(request)
            if not validation_result["valid"]:
                return SchedulingResult(success=False, error_message=validation_result["error"])
            
            # Check availability (simple mock)
            availability_result = await self._check_availability(request)
            if not availability_result["available"]:
                return SchedulingResult(success=False, error_message="Time slot not available")
            
            # Create calendar invite
            calendar_invite = self._create_calendar_invite(request, booking_id)
            
            # Send confirmation email
            email_result = await self._send_confirmation_email(request, booking_id, calendar_invite)
            
            # Return simple result
            result = SchedulingResult(
                success=True,
                booking_id=booking_id,
                confirmation_email_sent=email_result["success"],
                calendar_invite_sent=email_result["success"]
            )
            
            logger.info(f"Appointment scheduled successfully: {booking_id}")
            return result
            
        except Exception as e:
            logger.error(f"Scheduling error: {e}")
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
        """Create beautiful HTML email content with Claude theme"""
        appointment_time = request.to_datetime()
        formatted_date = appointment_time.strftime("%A, %B %d, %Y")
        formatted_time = appointment_time.strftime("%I:%M %p")
        service_type = request.service_type.title()
        timezone = request.timezone
        duration_minutes = request.duration_minutes
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Appointment Confirmation</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
        }}
        .header {{
            background: #D97706;
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}
        .content {{
            padding: 40px 30px;
        }}
        .appointment-details {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .next-steps h3 {{
            font-size: 18px;
            margin-bottom: 20px;
            color: #111827;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6b7280;
            font-size: 14px;
        }}
        .footer strong {{
            color: #111827;
        }}
        @media only screen and (max-width: 600px) {{
            .content {{
                padding: 30px 20px;
            }}
            .appointment-details {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <table align="center" cellpadding="0" cellspacing="0" border="0" style="margin: 0 auto 20px;">
                <tr>
                    <td align="center" valign="middle" style="background: #10b981; width: 60px; height: 60px; border-radius: 50%; font-size: 28px; color: #ffffff;">✓</td>
                </tr>
            </table>
            <h1 style="font-size: 24px; font-weight: 600; margin-bottom: 8px;">Appointment Confirmed</h1>
            <p style="opacity: 0.9; font-size: 16px;">Your {service_type} appointment is all set</p>
        </div>

        <div class="content">
            <div class="appointment-details">
                <table width="100%" cellpadding="0" cellspacing="0" style="font-size: 16px; color: #111827;">
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Service</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600;">{service_type}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Date</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600;">{formatted_date}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Time</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600;">{formatted_time} ({timezone})</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Duration</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600;">{duration_minutes} minutes</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Booking ID</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600; background: #e5e7eb; border-radius: 4px; font-family: monospace;">{booking_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 500; color: #6b7280;">Notes</td>
                        <td align="right" style="padding: 12px 0; font-weight: 600;">{request.notes or 'None'}</td>
                    </tr>
                </table>
            </div>

            <div class="next-steps">
                <h3>What's Next?</h3>
                <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                        <td valign="top" style="font-size: 18px; padding-right: 10px;">📅</td>
                        <td style="padding-bottom: 16px;">
                            <strong style="color: #111827;">Add to Calendar</strong><br>
                            Use the attached .ics file to add this appointment to your calendar.
                        </td>
                    </tr>
                    <tr>
                        <td valign="top" style="font-size: 18px; padding-right: 10px;">🔗</td>
                        <td style="padding-bottom: 16px;">
                            <strong style="color: #111827;">Meeting Link</strong><br>
                            You'll receive the meeting link 15 minutes before your appointment.
                        </td>
                    </tr>
                    <tr>
                        <td valign="top" style="font-size: 18px; padding-right: 10px;">📞</td>
                        <td>
                            <strong style="color: #111827;">Need Help?</strong><br>
                            Contact us at <a href="mailto:admin@kreyn.ai" style="color: #D97706; text-decoration: none;">admin@kreyn.ai</a>
                        </td>
                    </tr>
                </table>
            </div>

            <div style="text-align: center; margin: 30px 0;">
                <a href="mailto:admin@kreyn.ai" style="background: #D97706; color: #ffffff; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500;">Contact Support</a>
            </div>
        </div>

        <div class="footer">
            <p><strong>Kreyn AI</strong></p>
            <p>© 2025 Kreyn AI. All rights reserved.</p>
            <p>Need help? Email us at <a href="mailto:admin@kreyn.ai" style="color: #D97706; text-decoration: none;">admin@kreyn.ai</a></p>
        </div>
    </div>
</body>
</html>"""


# Global instance
_scheduling_service = None

def get_scheduling_service() -> SchedulingService:
    """Get the global scheduling service instance"""
    global _scheduling_service
    if _scheduling_service is None:
        _scheduling_service = SchedulingService()
    return _scheduling_service
