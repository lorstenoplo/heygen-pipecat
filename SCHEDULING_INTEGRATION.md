# Scheduling Integration Documentation

## Overview

The HeyGen-Pipecat conversational AI now includes comprehensive appointment scheduling functionality that follows the standardized protocol.py messaging system. This integration enables the AI to offer scheduling services, display booking interfaces, and send professional confirmation emails with calendar invites.

## Architecture

### Components

1. **Tool Calls** (`tools/scheduling_tools.py`) - Functions the LLM can call
2. **RTVI Actions** (`actions/rtvi_actions.py`) - Handlers for frontend interactions  
3. **Scheduling Service** (`services/scheduling_service.py`) - Core business logic
4. **Protocol Messages** (`messaging/protocol.py`) - Standardized messaging

### Flow Diagram

```
User Request → LLM → Tool Call → UI Popup → User Input → RTVI Action → Scheduling Service → Email + Calendar
```

## Implementation Details

### 1. Tool Calls (LLM Functions)

The LLM has access to three scheduling tools:

#### `show_scheduling_popup(service_type, context)`
- **Purpose**: Display scheduling interface to user
- **When to use**: User mentions booking, scheduling, or meeting requests
- **Parameters**:
  - `service_type`: "consultation", "demo", "onboarding", "support", "training"
  - `context`: Additional context about the appointment
- **Returns**: UI popup with booking form

#### `check_availability(date, time, service_type)`
- **Purpose**: Check if specific slot is available
- **Parameters**:
  - `date`: YYYY-MM-DD format
  - `time`: HH:MM format (24-hour)
  - `service_type`: Service type
- **Returns**: Availability status and reason if unavailable

#### `get_available_slots(date, service_type)`
- **Purpose**: Get all available time slots for a date
- **Parameters**:
  - `date`: YYYY-MM-DD format
  - `service_type`: Service type
- **Returns**: Array of available time slots

### 2. RTVI Action (Frontend Handler)

#### `schedule_appointment`
- **Service**: "scheduling"
- **Action**: "schedule_appointment"
- **Parameters**:
  - `email` (string, required)
  - `date` (string, required) - YYYY-MM-DD
  - `time` (string, required) - HH:MM
  - `service_type` (string) - Default: "consultation"
  - `duration_minutes` (number) - Default: 60
  - `timezone` (string) - Default: "UTC"
  - `notes` (string) - Optional

### 3. Protocol Messages

The system sends standardized messages throughout the booking process:

#### Process Flow Messages
- `PROCESS_START` - Booking process initiated
- `PROCESS_STEP` - Each step (validate, availability, calendar, email)
- `PROCESS_COMPLETE` - Successful booking
- `PROCESS_ERROR` - Any errors during booking

#### Tool Call Messages
- `TOOL_CALL_START` - Tool execution begins
- `TOOL_CALL_PROGRESS` - Progress updates
- `TOOL_CALL_RESULT` - Tool completion result
- `TOOL_CALL_ERROR` - Tool execution errors

#### UI Update Messages
- `UI_UPDATE` - Update scheduling popup
- `UI_COMPONENT_UPDATE` - Component-specific updates

## Frontend Integration

### 1. Scheduling Popup UI

When the LLM calls `show_scheduling_popup`, the frontend should display a booking form with:

```json
{
  "type": "ui_update",
  "component_id": "scheduling_popup",
  "ui_action": "show_popup",
  "payload": {
    "service_type": "consultation",
    "context": "Demo for new feature",
    "available_services": ["consultation", "demo", "onboarding", "support", "training"],
    "business_hours": "9 AM - 5 PM (Mon-Fri)",
    "timezone": "UTC"
  }
}
```

### 2. Form Submission

When user submits the form, send an RTVI action:

```javascript
// Frontend JavaScript
await rtvi.sendAction("scheduling", "schedule_appointment", {
  email: "user@example.com",
  date: "2025-07-15",
  time: "14:30",
  service_type: "consultation",
  duration_minutes: 60,
  timezone: "UTC",
  notes: "Looking forward to discussing the new features"
});
```

### 3. Real-time Updates

Listen for protocol messages to show booking progress:

```javascript
// Listen for process updates
rtvi.on('message', (message) => {
  if (message.type === 'process_step') {
    updateBookingProgress(message.payload);
  }
  
  if (message.type === 'process_complete') {
    showBookingConfirmation(message.payload);
  }
  
  if (message.type === 'ui_update' && message.component_id === 'booking_confirmation') {
    displayConfirmation(message.payload);
  }
});
```

## Email Integration

### Configuration

Set the Resend API key in your environment:

```bash
RESEND_API_KEY=your_resend_api_key_here
```

### Email Features

1. **Professional HTML Email**: Beautiful, responsive design
2. **Calendar Invite**: .ics file attachment for easy calendar import
3. **Booking Details**: All appointment information included
4. **Next Steps**: Clear instructions for the user

### Email Template

The system automatically sends emails with:
- Branded header with confirmation checkmark
- Detailed appointment card with all booking info
- Instructions for adding to calendar
- Professional footer with contact information

## API Reference

### Environment Variables

```bash
# Required for scheduling
RESEND_API_KEY=your_resend_key          # For email functionality

# Existing variables
OPENAI_API_KEY=your_openai_key
HEYGEN_API_KEY=your_heygen_key
DAILY_API_KEY=your_daily_key
# ... other existing keys
```

### Business Rules

- **Business Hours**: 9 AM - 5 PM, Monday through Friday
- **Time Zones**: UTC (configurable)
- **Booking Window**: Future dates only
- **Valid Services**: consultation, demo, onboarding, support, training
- **Default Duration**: 60 minutes
- **Time Slots**: 30-minute intervals

## Usage Examples

### 1. Basic Scheduling Request

**User**: "I'd like to book a demo"

**AI Response**: "I'd love to help you book a demo! Let me open up the scheduling interface for you."

*[AI calls `show_scheduling_popup("demo", "User interested in product demo")`]*

### 2. Specific Time Request

**User**: "Can I schedule a consultation for tomorrow at 2 PM?"

**AI**: "Let me check if 2 PM tomorrow is available for a consultation..."

*[AI calls `check_availability("2025-07-07", "14:00", "consultation")`]*

### 3. Date Availability

**User**: "What times are available on Friday?"

**AI**: "Let me pull up the available slots for Friday..."

*[AI calls `get_available_slots("2025-07-11", "consultation")`]*

## Error Handling

The system includes comprehensive error handling:

### Validation Errors
- Invalid email format
- Past dates/times
- Invalid service types
- Outside business hours
- Weekend bookings

### Service Errors
- Email service unavailable
- Calendar generation failure
- Database connectivity issues

### User Feedback
All errors are communicated through the protocol messaging system with clear, user-friendly messages.

## Testing

### 1. Tool Call Testing

```python
from tools.scheduling_tools import get_scheduling_tools

tools = get_scheduling_tools()

# Test popup display
result = await tools.show_scheduling_popup("demo", "Testing popup")
assert result["popup_shown"] == True

# Test availability check
result = await tools.check_availability("2025-07-15", "14:00", "consultation")
assert "available" in result
```

### 2. RTVI Action Testing

```python
from actions.rtvi_actions import handle_schedule_appointment

# Mock arguments
args = {
    "email": "test@example.com",
    "date": "2025-07-15",
    "time": "14:00",
    "service_type": "consultation"
}

result = await handle_schedule_appointment(None, None, args)
assert result["success"] == True
```

### 3. End-to-End Testing

1. Start conversation with AI
2. Request appointment booking
3. Verify popup display
4. Submit booking form
5. Check confirmation email
6. Verify calendar invite

## Deployment

### 1. Install Dependencies

```bash
pip install resend icalendar
```

### 2. Environment Setup

```bash
export RESEND_API_KEY=your_resend_key
```

### 3. Verify Integration

Check that all components are properly loaded:

```python
# In your application startup
from tools.scheduling_tools import get_scheduling_tools
from services.scheduling_service import get_scheduling_service

# Verify services are available
assert get_scheduling_tools() is not None
assert get_scheduling_service() is not None
```

## Monitoring

### Key Metrics to Track

1. **Booking Success Rate**: Percentage of successful bookings
2. **Email Delivery Rate**: Confirmation email success rate
3. **Calendar Invite Usage**: How many users add to calendar
4. **Error Rates**: Track validation and service errors
5. **Popular Time Slots**: Most requested booking times
6. **Service Type Distribution**: Which services are most popular

### Logging

The system logs all important events:

```python
# Booking initiated
logger.info(f"Booking request for {service_type} on {date} {time}")

# Email sent
logger.info(f"Confirmation email sent to {email}, ID: {email_id}")

# Errors
logger.error(f"Booking failed: {error_message}")
```

## Future Enhancements

### Potential Improvements

1. **Calendar Integration**: Direct integration with Google/Outlook calendars
2. **Timezone Support**: Automatic timezone detection and conversion
3. **Recurring Appointments**: Support for recurring bookings
4. **Cancellation/Rescheduling**: Allow users to modify bookings
5. **SMS Notifications**: Text message confirmations and reminders
6. **Payment Integration**: Collect payment during booking process
7. **Resource Management**: Room/equipment booking
8. **Waitlist Management**: Queue users for popular time slots

### Scalability Considerations

- **Database Integration**: Replace mock availability checks with real calendar system
- **Queue System**: Handle high-volume booking requests
- **Rate Limiting**: Prevent spam bookings
- **Multi-tenant Support**: Support multiple organizations
- **API Rate Limits**: Respect email service limits

This scheduling integration provides a solid foundation for appointment booking while maintaining the conversational, natural feel of the AI interaction.
