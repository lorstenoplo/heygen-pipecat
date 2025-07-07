"""
Simple scheduling service with email and calendar integration.
"""

import uuid
from datetime import datetime, timedelta
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
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Appointment Confirmation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.5;
            color: #1f2937;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #fef3e2 0%, #fff7ed 100%);
        }}
        
        .container {{
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #fed7aa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            padding: 40px 32px 32px 32px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        
        .success-icon {{
            width: 56px;
            height: 56px;
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            font-size: 28px;
            color: white;
            backdrop-filter: blur(10px);
        }}
        
        .content {{
            padding: 36px 32px;
        }}
        
        .appointment-card {{
            background: linear-gradient(135deg, #fefbf3 0%, #fef7ed 100%);
            border: 1px solid #fed7aa;
            border-radius: 8px;
            padding: 28px;
            margin: 28px 0;
            position: relative;
            overflow: hidden;
        }}
        
        .appointment-card::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, rgba(249,115,22,0.05) 0%, transparent 100%);
            border-radius: 50%;
            transform: translate(30px, -30px);
        }}
        
        .card-content {{
            position: relative;
            z-index: 1;
        }}
        
        .section-title {{
            color: #111827;
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 20px 0;
            display: flex;
            align-items: center;
        }}
        
        .title-accent {{
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            border-radius: 2px;
            margin-right: 12px;
        }}
        
        .detail-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin: 18px 0;
            padding: 14px 0;
            border-bottom: 1px solid rgba(249,115,22,0.1);
            gap: 16px;
        }}
        
        .detail-row:last-child {{
            border-bottom: none;
        }}
        
        .detail-label {{
            font-weight: 500;
            color: #6b7280;
            font-size: 14px;
            flex-shrink: 0;
            min-width: 80px;
        }}
        
        .detail-value {{
            color: #111827;
            font-weight: 600;
            text-align: right;
            word-break: break-word;
            flex: 1;
        }}
        
        .booking-id {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            background: linear-gradient(135deg, rgba(249,115,22,0.1) 0%, rgba(234,88,12,0.05) 100%);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid rgba(249,115,22,0.2);
            word-break: break-all;
        }}
        
        .tips-list {{
            margin: 24px 0;
            padding: 0;
            list-style: none;
        }}
        
        .tips-list li {{
            margin: 16px 0;
            padding: 16px;
            display: flex;
            align-items: flex-start;
            font-size: 15px;
            background: linear-gradient(135deg, rgba(17,24,39,0.02) 0%, rgba(17,24,39,0.01) 100%);
            border-radius: 8px;
            border-left: 3px solid #f97316;
        }}
        
        .tips-list li .icon {{
            margin-right: 16px;
            font-size: 18px;
            margin-top: 2px;
            flex-shrink: 0;
        }}
        
        .cta-button {{
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: #ffffff;
            padding: 14px 32px;
            text-decoration: none;
            border-radius: 8px;
            display: inline-block;
            font-weight: 600;
            font-size: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
            border: 1px solid #374151;
        }}
        
        .cta-button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}
        
        .info-box {{
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fbbf24;
            border-radius: 8px;
            padding: 20px;
            margin: 28px 0;
            border-left: 4px solid #f59e0b;
            position: relative;
            overflow: hidden;
        }}
        
        .info-box::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, transparent 100%);
            border-radius: 50%;
            transform: translate(20px, -20px);
        }}
        
        .info-content {{
            position: relative;
            z-index: 1;
        }}
        
        .footer {{
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            padding: 32px;
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            position: relative;
            overflow: hidden;
        }}
        
        .footer::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(249,115,22,0.05) 0%, transparent 100%);
        }}
        
        .footer-content {{
            position: relative;
            z-index: 1;
        }}
        
        h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        
        .subtitle {{
            margin: 12px 0 0 0;
            color: rgba(255,255,255,0.9);
            font-size: 16px;
            font-weight: 400;
        }}
        
        a {{
            color: #f97316;
            text-decoration: none;
            font-weight: 500;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .footer p {{
            margin: 6px 0;
        }}
        
        .footer strong {{
            color: #f97316;
            font-size: 16px;
        }}
        
        .footer .support-text {{
            color: #9ca3af;
        }}
        
        /* Mobile Responsive Styles */
        @media only screen and (max-width: 600px) {{
            body {{
                padding: 10px;
                font-size: 14px;
            }}
            
            .header {{
                padding: 24px 16px 20px 16px;
            }}
            
            .success-icon {{
                width: 48px;
                height: 48px;
                font-size: 24px;
                margin-bottom: 16px;
            }}
            
            h1 {{
                font-size: 22px;
                line-height: 1.3;
            }}
            
            .subtitle {{
                font-size: 14px;
                line-height: 1.4;
            }}
            
            .content {{
                padding: 20px 16px;
            }}
            
            .appointment-card {{
                padding: 20px 16px;
                margin: 20px 0;
            }}
            
            .section-title {{
                font-size: 16px;
                margin-bottom: 16px;
                flex-wrap: wrap;
            }}
            
            .title-accent {{
                width: 3px;
                height: 16px;
                margin-right: 8px;
            }}
            
            .detail-row {{
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
                margin: 12px 0;
                padding: 12px 0;
            }}
            
            .detail-label {{
                font-size: 13px;
                margin-bottom: 4px;
                min-width: auto;
            }}
            
            .detail-value {{
                text-align: left;
                font-size: 14px;
                font-weight: 600;
                width: 100%;
            }}
            
            .booking-id {{
                font-size: 12px;
                padding: 8px 12px;
                word-break: break-all;
                line-height: 1.3;
            }}
            
            .tips-list li {{
                padding: 12px;
                font-size: 14px;
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .tips-list li .icon {{
                margin-right: 0;
                margin-bottom: 8px;
                font-size: 16px;
            }}
            
            .cta-button {{
                padding: 12px 24px;
                font-size: 14px;
                width: auto;
                display: inline-block;
            }}
            
            .info-box {{
                padding: 16px;
                margin: 20px 0;
            }}
            
            .info-content {{
                font-size: 14px;
                line-height: 1.4;
            }}
            
            .footer {{
                padding: 24px 16px;
                font-size: 13px;
            }}
            
            .footer strong {{
                font-size: 14px;
            }}
        }}
        
        /* Extra small screens */
        @media only screen and (max-width: 400px) {{
            body {{
                padding: 8px;
            }}
            
            .container {{
                border-radius: 8px;
            }}
            
            .header {{
                padding: 20px 12px 16px 12px;
            }}
            
            .success-icon {{
                width: 44px;
                height: 44px;
                font-size: 22px;
            }}
            
            h1 {{
                font-size: 20px;
            }}
            
            .content {{
                padding: 16px 12px;
            }}
            
            .appointment-card {{
                padding: 16px 12px;
            }}
            
            .section-title {{
                font-size: 15px;
            }}
            
            .detail-value {{
                font-size: 13px;
            }}
            
            .booking-id {{
                font-size: 11px;
                padding: 6px 10px;
            }}
            
            .tips-list li {{
                padding: 10px;
                font-size: 13px;
            }}
            
            .cta-button {{
                padding: 10px 20px;
                font-size: 13px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <div class="success-icon">✓</div>
                <h1>Appointment Confirmed</h1>
                <p class="subtitle">Your {request.service_type.title()} appointment has been successfully scheduled</p>
            </div>
        </div>
        
        <div class="content">
            <div class="appointment-card">
                <div class="card-content">
                    <h3 class="section-title">
                        <span class="title-accent"></span>
                        Appointment Details
                    </h3>
                    
                    <div class="detail-row">
                        <span class="detail-label">Service</span>
                        <span class="detail-value">{request.service_type.title()}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Date</span>
                        <span class="detail-value">{formatted_date}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Time</span>
                        <span class="detail-value">{formatted_time} ({request.timezone})</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Duration</span>
                        <span class="detail-value">{request.duration_minutes} minutes</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Booking ID</span>
                        <span class="detail-value booking-id">{booking_id}</span>
                    </div>
                    
                    {f'<div class="detail-row"><span class="detail-label">Notes</span><span class="detail-value">{request.notes}</span></div>' if request.notes else ''}
                </div>
            </div>
            
            <div style="margin: 32px 0;">
                <h3 class="section-title">
                    <span class="title-accent"></span>
                    What's Next?
                </h3>
                <ul class="tips-list">
                    <li>
                        <span class="icon">📅</span>
                        <div><strong>Add to Calendar:</strong> <span style="color: #4b5563;">Click the attached calendar file (.ics) to add this appointment to your calendar</span></div>
                    </li>
                    <li>
                        <span class="icon">🔗</span>
                        <div><strong>Join Link:</strong> <span style="color: #4b5563;">We'll send you the meeting link 15 minutes before the appointment</span></div>
                    </li>
                    <li>
                        <span class="icon">📞</span>
                        <div><strong>Need Changes?:</strong> <span style="color: #4b5563;">Reply to this email or contact us at admin@kreyn.ai to reschedule</span></div>
                    </li>
                </ul>
            </div>
            
            <div style="text-align: center; margin: 32px 0;">
                <a href="mailto:admin@kreyn.ai" class="cta-button">Contact Support</a>
            </div>
            
            <div class="info-box">
                <div class="info-content">
                    <strong style="color: #92400e;">💡 Pro Tip:</strong> <span style="color: #78350f;">Save this email for your records and join the meeting 5 minutes early for the best experience.</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="footer-content">
                <p><strong>Thank you for choosing Kreyn AI</strong></p>
                <p class="support-text">© 2025 Kreyn AI. All rights reserved.</p>
                <p class="support-text">For support, contact us at <a href="mailto:admin@kreyn.ai">admin@kreyn.ai</a></p>
            </div>
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
