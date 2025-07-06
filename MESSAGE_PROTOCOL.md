# Standard Message Protocol Documentation

## Overview

The Standard Message Protocol provides a structured, extensible way to handle complex multi-step operations, tool calls, and UI updates in the HeyGen-Pipecat conversational AI system. This protocol enables sophisticated workflows like booking processes, scheduling operations, and complex agent interactions.

## Core Components

### 1. Message Types (`MessageType` Enum)

```python
class MessageType(str, Enum):
    # Tool/Function related
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_PROGRESS = "tool_call_progress" 
    TOOL_CALL_RESULT = "tool_call_result"
    TOOL_CALL_ERROR = "tool_call_error"
    
    # UI Updates
    UI_UPDATE = "ui_update"
    UI_COMPONENT_UPDATE = "ui_component_update"
    UI_FOLLOW_UP = "ui_follow_up"
    
    # Multi-step process
    PROCESS_START = "process_start"
    PROCESS_STEP = "process_step"
    PROCESS_COMPLETE = "process_complete"
    PROCESS_ERROR = "process_error"
    
    # Agent state
    AGENT_THINKING = "agent_thinking"
    AGENT_SPEAKING = "agent_speaking"
    AGENT_WAITING = "agent_waiting"
    
    # System
    SYSTEM_STATUS = "system_status"
    SYSTEM_ERROR = "system_error"
```

### 2. StandardMessage Structure

```python
@dataclass
class StandardMessage:
    type: MessageType
    timestamp: Optional[str] = None
    message_id: Optional[str] = None
    
    # Core identifiers
    tool_call_id: Optional[str] = None
    process_id: Optional[str] = None
    component_id: Optional[str] = None
    
    # Message metadata
    name: Optional[str] = None
    priority: Priority = Priority.NORMAL
    expires_at: Optional[str] = None
    
    # Content
    payload: Optional[Dict[str, Any]] = None
    
    # Contextual info
    tool_call: Optional[ToolCallInfo] = None
    process_step: Optional[ProcessStep] = None
    
    # UI specific
    ui_target: Optional[str] = None
    ui_action: Optional[str] = None
```

## Usage Patterns

### 1. Simple Follow-up Questions

```python
from services.message_service import get_message_service

message_service = get_message_service()

await message_service.send_follow_up_questions([
    "What time works best for you?",
    "Should I set a reminder?",
    "Would you like to make this recurring?"
])
```

### 2. Tool Call Lifecycle

```python
# Start a tool call
tool_id = await message_service.start_tool_call(
    tool_name="calendar_scheduler",
    parameters={"date": "2024-01-15", "time": "14:00"}
)

# Update progress
await message_service.update_tool_call(
    tool_id=tool_id,
    progress={"status": "checking_availability", "progress": 50}
)

# Complete with result
await message_service.complete_tool_call(
    tool_id=tool_id,
    result={"scheduled": True, "confirmation_id": "ABC123"}
)
```

### 3. Multi-step Process

```python
# Start a complex process
process_id = await message_service.start_process(
    name="Hotel Booking",
    total_steps=4,
    description="Complete hotel booking with payment"
)

# Update each step
await message_service.update_process_step(
    process_id=process_id,
    step_name="Checking Availability",
    status="running"
)

await message_service.update_process_step(
    process_id=process_id,
    step_name="Checking Availability", 
    status="completed",
    data={"rooms_available": 5, "price": 150.00}
)

# Complete the process
await message_service.complete_process(
    process_id=process_id,
    result={"booking_id": "BOOK123", "total_cost": 450.00}
)
```

### 4. Agent State Management

```python
# Indicate agent is thinking
await message_service.set_agent_state("thinking", {
    "context": "Processing complex scheduling request"
})

# Indicate agent is speaking
await message_service.set_agent_state("speaking", {
    "message": "Let me check your calendar..."
})

# Indicate agent is waiting
await message_service.set_agent_state("waiting", {
    "waiting_for": "user_confirmation"
})
```

### 5. Complex Workflow Example

```python
async def booking_workflow(booking_details):
    # Start the overall process
    process_id = await message_service.start_process(
        name="Flight Booking",
        total_steps=5,
        description="Complete flight booking with seat selection"
    )
    
    try:
        # Step 1: Search flights
        search_tool = await message_service.start_tool_call(
            tool_name="flight_search",
            parameters=booking_details
        )
        
        await message_service.update_process_step(
            process_id=process_id,
            step_name="Searching Flights",
            status="running"
        )
        
        # Simulate search progress
        await message_service.update_tool_call(
            search_tool,
            {"status": "searching", "airlines_checked": 3}
        )
        
        # Complete search
        search_results = {"flights": [...], "best_price": 450}
        await message_service.complete_tool_call(search_tool, search_results)
        
        await message_service.update_process_step(
            process_id=process_id,
            step_name="Searching Flights",
            status="completed",
            data=search_results
        )
        
        # Step 2: Seat selection
        seat_tool = await message_service.start_tool_call(
            tool_name="seat_selector",
            parameters={"flight_id": "FL123", "preferences": "window"}
        )
        
        await message_service.update_process_step(
            process_id=process_id,
            step_name="Selecting Seats",
            status="running"
        )
        
        seat_result = {"seat": "12A", "cost": 25}
        await message_service.complete_tool_call(seat_tool, seat_result)
        
        await message_service.update_process_step(
            process_id=process_id,
            step_name="Selecting Seats", 
            status="completed",
            data=seat_result
        )
        
        # Continue with payment, confirmation, etc...
        
        # Final completion
        await message_service.complete_process(
            process_id=process_id,
            result={
                "booking_reference": "ABC123",
                "total_cost": 475,
                "confirmation_sent": True
            }
        )
        
        # Send relevant follow-ups
        await message_service.send_follow_up_questions([
            "Would you like travel insurance?",
            "Should I add this to your calendar?",
            "Do you need a rental car?"
        ])
        
    except Exception as e:
        await message_service.error_process(process_id, str(e))
```

## Message Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent/LLM     │    │ Message Service │    │   RTVI/UI       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Tool Calls  │─┼────┼─│ Protocol    │─┼────┼─│ UI Updates  │ │
│ │ Processes   │ │    │ │ Conversion  │ │    │ │ Progress    │ │
│ │ State       │ │    │ │ Routing     │ │    │ │ Indicators  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Benefits

1. **Standardization**: Consistent message format across all operations
2. **Traceability**: Every operation has unique IDs for tracking
3. **Progress Tracking**: Real-time updates on long-running operations  
4. **Error Handling**: Structured error reporting and recovery
5. **UI Integration**: Seamless UI updates and user feedback
6. **Extensibility**: Easy to add new message types and workflows
7. **State Management**: Clear tracking of agent and process states

## Integration with Existing System

The message protocol integrates seamlessly with the existing RTVI system:

1. **Message Service**: Handles protocol conversion and routing
2. **RTVI Integration**: Converts StandardMessage to RTVIServerMessageFrame
3. **Event Handlers**: Updated to use message service for follow-ups
4. **Pipeline Manager**: Coordinates message service with pipeline state

## Best Practices

1. **Always use unique IDs**: Let the system generate IDs automatically
2. **Include meaningful progress updates**: Keep users informed during long operations
3. **Handle errors gracefully**: Use error messages to provide actionable feedback
4. **Set appropriate priorities**: Use HIGH priority for critical errors
5. **Clean up resources**: Use the cleanup methods for long-running services
6. **Provide context**: Include relevant data in progress updates

## Error Handling

```python
try:
    tool_id = await message_service.start_tool_call("complex_operation", params)
    
    # ... operation logic ...
    
    await message_service.complete_tool_call(tool_id, result)
    
except ValidationError as e:
    await message_service.error_tool_call(tool_id, f"Validation failed: {e}")
except NetworkError as e:
    await message_service.error_tool_call(tool_id, f"Network error: {e}")
except Exception as e:
    await message_service.error_tool_call(tool_id, f"Unexpected error: {e}")
```

This protocol provides a robust foundation for complex conversational AI interactions while maintaining clarity and extensibility.
