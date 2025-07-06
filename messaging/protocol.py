"""
Standard message protocol for GenUI platform supporting complex multi-step tool calls.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import uuid


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


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ToolCallInfo:
    """Information about a tool call"""
    id: str
    name: str
    parameters: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None  # For nested tool calls
    

@dataclass
class ProcessStep:
    """Information about a process step"""
    step_id: str
    step_name: str
    step_index: int
    total_steps: int
    status: str  # pending, running, completed, failed
    description: Optional[str] = None
    

@dataclass
class StandardMessage:
    """Standard message format for GenUI platform"""
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
    ui_target: Optional[str] = None  # Which UI component to update
    ui_action: Optional[str] = None  # What action to take
    
    def __post_init__(self):
        if self.payload is None:
            self.payload = {}
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Clean up None values for cleaner JSON
        return {k: v for k, v in result.items() if v is not None}
    
    def to_rtvi_frame_data(self) -> Dict[str, Any]:
        """Convert to RTVI frame data format"""
        return {
            "type": self.type.value,
            "payload": self.to_dict()
        }


# Helper functions for common message types
def create_tool_call_start(tool_name: str, tool_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> StandardMessage:
    if tool_id is None:
        tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    
    return StandardMessage(
        type=MessageType.TOOL_CALL_START,
        tool_call_id=tool_id,
        name=tool_name,
        tool_call=ToolCallInfo(
            id=tool_id,
            name=tool_name,
            parameters=parameters or {}
        ),
        payload={"status": "starting", "parameters": parameters or {}}
    )


def create_tool_call_progress(tool_id: str, progress: Dict[str, Any]) -> StandardMessage:
    return StandardMessage(
        type=MessageType.TOOL_CALL_PROGRESS,
        tool_call_id=tool_id,
        payload=progress
    )


def create_tool_call_result(tool_id: str, result: Any, tool_name: Optional[str] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.TOOL_CALL_RESULT,
        tool_call_id=tool_id,
        name=tool_name,
        payload={"result": result, "status": "completed"}
    )


def create_tool_call_error(tool_id: str, error: str, tool_name: Optional[str] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.TOOL_CALL_ERROR,
        tool_call_id=tool_id,
        name=tool_name,
        priority=Priority.HIGH,
        payload={"error": error, "status": "failed"}
    )


def create_process_start(process_id: Optional[str] = None, name: Optional[str] = None, 
                        total_steps: Optional[int] = None, description: Optional[str] = None) -> StandardMessage:
    if process_id is None:
        process_id = f"proc_{uuid.uuid4().hex[:8]}"
    
    return StandardMessage(
        type=MessageType.PROCESS_START,
        process_id=process_id,
        name=name,
        payload={
            "total_steps": total_steps,
            "description": description,
            "status": "started"
        }
    )


def create_process_step(process_id: str, step_id: str, step_name: str, 
                       step_index: int, total_steps: int, 
                       status: str = "running", data: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.PROCESS_STEP,
        process_id=process_id,
        name=step_name,
        process_step=ProcessStep(
            step_id=step_id,
            step_name=step_name,
            step_index=step_index,
            total_steps=total_steps,
            status=status,
            description=description
        ),
        payload=data or {}
    )


def create_process_complete(process_id: str, name: Optional[str] = None, 
                           result: Optional[Dict[str, Any]] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.PROCESS_COMPLETE,
        process_id=process_id,
        name=name,
        payload={
            "result": result or {},
            "status": "completed"
        }
    )


def create_process_error(process_id: str, error: str, name: Optional[str] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.PROCESS_ERROR,
        process_id=process_id,
        name=name,
        priority=Priority.HIGH,
        payload={"error": error, "status": "failed"}
    )


def create_ui_update(component_id: str, action: str, data: Dict[str, Any]) -> StandardMessage:
    return StandardMessage(
        type=MessageType.UI_UPDATE,
        component_id=component_id,
        ui_target=component_id,
        ui_action=action,
        payload=data
    )


def create_follow_up_questions(questions: List[str], context: Optional[str] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.UI_FOLLOW_UP,
        name="follow_up_questions",
        component_id="follow_up_panel",
        ui_target="follow_up_panel",
        ui_action="update_questions",
        payload={
            "questions": questions,
            "context": context,
            "ui_component": "follow_up_panel"
        }
    )


def create_agent_state(state: str, details: Optional[Dict[str, Any]] = None) -> StandardMessage:
    """Create agent state message (thinking, speaking, waiting)"""
    message_type = getattr(MessageType, f"AGENT_{state.upper()}", MessageType.AGENT_WAITING)
    
    return StandardMessage(
        type=message_type,
        name=f"agent_{state}",
        payload=details or {"state": state}
    )


def create_system_status(status: str, details: Optional[Dict[str, Any]] = None) -> StandardMessage:
    return StandardMessage(
        type=MessageType.SYSTEM_STATUS,
        name="system_status",
        payload={"status": status, **(details or {})}
    )


# Complex multi-step workflow helpers
def create_scheduler_flow_messages(schedule_id: Optional[str] = None, tasks: Optional[List[Dict[str, Any]]] = None) -> List[StandardMessage]:
    """Example of creating messages for a scheduler flow"""
    if schedule_id is None:
        schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"
    
    if tasks is None:
        tasks = []
    
    messages = []
    
    # Start process
    messages.append(create_process_start(
        process_id=schedule_id,
        name="schedule_creation",
        total_steps=len(tasks),
        description="Creating schedule with multiple tasks"
    ))
    
    # Each task as a step
    for i, task in enumerate(tasks):
        messages.append(create_process_step(
            process_id=schedule_id,
            step_id=f"task_{i}",
            step_name=f"Schedule {task.get('name', f'Task {i+1}')}",
            step_index=i,
            total_steps=len(tasks),
            status="pending",
            data=task,
            description=task.get('description')
        ))
    
    return messages


def create_booking_flow_messages(booking_id: Optional[str] = None) -> List[StandardMessage]:
    """Example booking flow with multiple steps"""
    if booking_id is None:
        booking_id = f"booking_{uuid.uuid4().hex[:8]}"
    
    steps = [
        {"name": "Check Availability", "description": "Checking available time slots"},
        {"name": "Validate Details", "description": "Validating booking information"},
        {"name": "Process Payment", "description": "Processing payment information"},
        {"name": "Confirm Booking", "description": "Confirming and sending confirmation"}
    ]
    
    messages = []
    
    # Start process
    messages.append(create_process_start(
        process_id=booking_id,
        name="booking_process",
        total_steps=len(steps),
        description="Processing booking request"
    ))
    
    # Add steps
    for i, step in enumerate(steps):
        messages.append(create_process_step(
            process_id=booking_id,
            step_id=f"step_{i}",
            step_name=step["name"],
            step_index=i,
            total_steps=len(steps),
            status="pending",
            description=step["description"]
        ))
    
    return messages
