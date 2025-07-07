# 📅 Simple Scheduling Integration Guide

## Overview

This document explains the simplified scheduling integration in the HeyGen-Pipecat conversational AI system. The integration follows a clean separation between AI conversation handling and UI data collection.

## Architecture Flow

```
User Request → LLM Tool Call → Frontend Popup → RTVI Action → Email Sent
```

### 1. **User Request**
User expresses interest in scheduling (e.g., "I'd like to book a demo")

### 2. **LLM Tool Call** 
AI calls `show_scheduling_popup` tool (does NOT collect date/time/email)

### 3. **Frontend Popup**
UI displays scheduling form for user to fill out

### 4. **RTVI Action**
Frontend sends data via `scheduling.schedule_appointment` action

### 5. **Email Sent**
System processes appointment and sends confirmation email with calendar invite

## 🛠️ Implementation Details

### LLM Tool Available

#### `show_scheduling_popup` (Only Tool)
- **Purpose**: Opens scheduling UI popup
- **When to use**: When user wants to schedule anything
- **Parameters**:
  - `service_type`: consultation, demo, onboarding, support, training
  - `context`: Brief description (optional)

> **Note**: Availability checking and slot listing tools have been removed for simplicity. The popup handles all scheduling logic.

### RTVI Action Available

#### `scheduling.schedule_appointment`
- **Purpose**: Process scheduling data from frontend
- **Parameters**:
  - `email` (required): User's email
  - `date` (required): YYYY-MM-DD format
  - `time` (required): HH:MM format
  - `service_type`: Type of service
  - `duration_minutes`: Duration (default: 60)
  - `timezone`: User timezone (default: UTC)
  - `notes`: Additional notes

## 🎯 Key Rules for AI Behavior

### ✅ DO:
- Call `show_scheduling_popup` when user wants to schedule
- Let the popup handle all data collection
- Be conversational and helpful
- Offer scheduling proactively when relevant

### ❌ DON'T:
- Ask for date/time/email through conversation
- Try to collect scheduling details via voice/chat
- Overcomplicate the process

## 📨 Email Integration

### Features:
- Beautiful HTML email templates with Claude theme colors
- Modern rounded design with orange gradient
- Calendar invite (.ics) attachment
- Professional branding
- Booking confirmation details
- Clean, readable layout

### Configuration:
- Requires `RESEND_API_KEY` in environment
- Uses `admin@kreyn.ai` as sender
- Includes booking ID for tracking

## 🔧 Configuration Required

### Environment Variables:
```bash
RESEND_API_KEY=your_resend_api_key_here
```

### Dependencies:
```bash
resend>=2.0.0
icalendar>=6.0.0
```

## 🚀 Frontend Integration

### 1. Listen for UI Updates
```javascript
// Listen for scheduling popup requests
rtvi.on('message', (message) => {
  if (message.type === 'ui_update' && 
      message.component_id === 'scheduling_popup' &&
      message.action === 'show_popup') {
    // Show your scheduling form
    showSchedulingPopup(message.data);
  }
});
```

### 2. Send Scheduling Data
```javascript
// When user submits form
rtvi.action({
  service: 'scheduling',
  action: 'schedule_appointment',
  arguments: {
    email: userEmail,
    date: selectedDate,
    time: selectedTime,
    service_type: serviceType,
    timezone: userTimezone,
    notes: userNotes
  }
});
```

### 3. Handle Response
```javascript
// Handle scheduling result
rtvi.on('actionResponse', (response) => {
  if (response.action === 'schedule_appointment') {
    if (response.result.success) {
      // Show success message
      showSuccessMessage(response.result.booking_id);
    } else {
      // Show error message
      showErrorMessage(response.result.error);
    }
  }
});
```

## 🎨 UI Components Expected

### Scheduling Popup Should Include:
- Service type selection (pre-filled from tool call)
- Date picker (business days only)
- Time slot picker (9 AM - 5 PM)
- Email input field
- Timezone selector
- Notes text area (optional)
- Submit/Cancel buttons

### Business Rules:
- Monday-Friday only
- 9 AM - 5 PM business hours
- 30-minute time slots
- 60-minute default duration

## 🧪 Testing

### Manual Test Flow:
1. Say "I'd like to book a demo"
2. AI should call `show_scheduling_popup` tool
3. Popup should appear with demo pre-selected
4. Fill out form and submit
5. Check for confirmation email
6. Verify calendar invite attachment

### Test Data:
```javascript
{
  email: "test@example.com",
  date: "2025-07-07",
  time: "14:00", 
  service_type: "demo",
  timezone: "UTC",
  notes: "Looking forward to it!"
}
```

## 🔍 Troubleshooting

### Common Issues:

1. **Email not sending**: Check `RESEND_API_KEY` configuration
2. **Calendar invite missing**: Install `icalendar` dependency
3. **Popup not showing**: Verify RTVI message handling
4. **Time validation errors**: Ensure business hours compliance

### Debug Messages:
- All scheduling operations log to console with `logger.info`
- Email sending includes response IDs
- Error handling provides detailed messages

## 📚 Example Usage

### Conversation Examples:

**User**: "Can I schedule a consultation?"
**AI**: *Calls show_scheduling_popup tool*
**Result**: Popup appears with consultation pre-selected

**User**: "I need to book a demo for next week"
**AI**: *Calls show_scheduling_popup with service_type="demo"*
**Result**: Popup appears ready for date/time selection

## 🎯 Success Metrics

A successful integration should achieve:
- ✅ Smooth conversation flow without data collection
- ✅ Intuitive popup UI that's easy to use
- ✅ Reliable email delivery with calendar invites
- ✅ Clear confirmation and error messaging
- ✅ Professional appearance and branding

---

*This simplified integration provides a seamless scheduling experience that leverages AI for natural conversation while maintaining a clean UI for data collection.*
