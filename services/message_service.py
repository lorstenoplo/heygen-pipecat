"""
Message service for sending standardized messages through the RTVI system.
"""

from typing import Optional, List, Dict, Any
from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from ..messaging.protocol import (
    StandardMessage, MessageType, Priority,
    create_follow_up_questions, create_tool_call_start, create_tool_call_progress,
    create_tool_call_result, create_tool_call_error, create_process_start,
    create_process_step, create_process_complete, create_process_error,
    create_agent_state, create_ui_update
)


class MessageService:
    """Service for sending standardized messages through RTVI."""
    
    def __init__(self, rtvi_processor=None):
        self.rtvi_processor = rtvi_processor
        self._active_processes = {}
        self._active_tool_calls = {}
    
    def set_rtvi_processor(self, processor):
        """Set the RTVI processor for sending messages."""
        self.rtvi_processor = processor
    
    async def send_message(self, message: StandardMessage) -> bool:
        """Send a standard message through RTVI."""
        if not self.rtvi_processor:
            logger.warning("No RTVI processor available for sending message")
            return False
        
        try:
            frame_data = message.to_rtvi_frame_data()
            frame = RTVIServerMessageFrame(data=frame_data)
            await self.rtvi_processor.push_frame(frame)
            
            logger.debug(f"Sent message: {message.type.value} (ID: {message.message_id})")
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    # Follow-up questions
    async def send_follow_up_questions(self, questions: List[str], context: Optional[str] = None) -> bool:
        """Send follow-up questions to the UI."""
        message = create_follow_up_questions(questions, context)
        return await self.send_message(message)
    
    # Tool call lifecycle
    async def start_tool_call(self, tool_name: str, parameters: Optional[Dict[str, Any]] = None, tool_id: Optional[str] = None) -> str:
        """Start a tool call and return the tool ID."""
        message = create_tool_call_start(tool_name, tool_id, parameters)
        tool_id = message.tool_call_id
        
        if tool_id:  # Ensure tool_id is not None
            self._active_tool_calls[tool_id] = {
                "name": tool_name,
                "status": "running",
                "start_time": message.timestamp
            }
        
        await self.send_message(message)
        return tool_id or ""  # Return empty string if None
    
    async def update_tool_call(self, tool_id: str, progress: Dict[str, Any]) -> bool:
        """Send progress update for a tool call."""
        if tool_id not in self._active_tool_calls:
            logger.warning(f"Tool call {tool_id} not found in active calls")
            return False
        
        message = create_tool_call_progress(tool_id, progress)
        return await self.send_message(message)
    
    async def complete_tool_call(self, tool_id: str, result: Any) -> bool:
        """Complete a tool call with result."""
        if tool_id not in self._active_tool_calls:
            logger.warning(f"Tool call {tool_id} not found in active calls")
            return False
        
        tool_info = self._active_tool_calls.pop(tool_id)
        message = create_tool_call_result(tool_id, result, tool_info["name"])
        return await self.send_message(message)
    
    async def error_tool_call(self, tool_id: str, error: str) -> bool:
        """Mark a tool call as failed with error."""
        if tool_id not in self._active_tool_calls:
            logger.warning(f"Tool call {tool_id} not found in active calls")
            return False
        
        tool_info = self._active_tool_calls.pop(tool_id)
        message = create_tool_call_error(tool_id, error, tool_info["name"])
        return await self.send_message(message)
    
    # Process lifecycle
    async def start_process(self, name: str, total_steps: Optional[int] = None, 
                           description: Optional[str] = None, process_id: Optional[str] = None) -> str:
        """Start a multi-step process and return process ID."""
        message = create_process_start(process_id, name, total_steps, description)
        process_id = message.process_id
        
        if process_id:  # Ensure process_id is not None
            self._active_processes[process_id] = {
                "name": name,
                "total_steps": total_steps,
                "current_step": 0,
                "status": "running",
                "start_time": message.timestamp
            }
        
        await self.send_message(message)
        return process_id or ""  # Return empty string if None
    
    async def update_process_step(self, process_id: str, step_name: str, 
                                 status: str = "running", data: Optional[Dict[str, Any]] = None,
                                 description: Optional[str] = None) -> bool:
        """Update the current step of a process."""
        if process_id not in self._active_processes:
            logger.warning(f"Process {process_id} not found in active processes")
            return False
        
        process_info = self._active_processes[process_id]
        step_index = process_info["current_step"]
        total_steps = process_info["total_steps"] or 1
        
        message = create_process_step(
            process_id=process_id,
            step_id=f"step_{step_index}",
            step_name=step_name,
            step_index=step_index,
            total_steps=total_steps,
            status=status,
            data=data,
            description=description
        )
        
        if status in ["completed", "failed"]:
            process_info["current_step"] += 1
        
        return await self.send_message(message)
    
    async def complete_process(self, process_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Complete a process with final result."""
        if process_id not in self._active_processes:
            logger.warning(f"Process {process_id} not found in active processes")
            return False
        
        process_info = self._active_processes.pop(process_id)
        message = create_process_complete(process_id, process_info["name"], result)
        return await self.send_message(message)
    
    async def error_process(self, process_id: str, error: str) -> bool:
        """Mark a process as failed with error."""
        if process_id not in self._active_processes:
            logger.warning(f"Process {process_id} not found in active processes")
            return False
        
        process_info = self._active_processes.pop(process_id)
        message = create_process_error(process_id, error, process_info["name"])
        return await self.send_message(message)
    
    # Agent state
    async def set_agent_state(self, state: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Set agent state (thinking, speaking, waiting)."""
        message = create_agent_state(state, details)
        return await self.send_message(message)
    
    # UI updates
    async def update_ui_component(self, component_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Update a UI component."""
        message = create_ui_update(component_id, action, data)
        return await self.send_message(message)
    
    # Utility methods
    def get_active_tool_calls(self) -> Dict[str, Any]:
        """Get all active tool calls."""
        return self._active_tool_calls.copy()
    
    def get_active_processes(self) -> Dict[str, Any]:
        """Get all active processes."""
        return self._active_processes.copy()
    
    async def cleanup_stale_operations(self, max_age_minutes: int = 30):
        """Clean up stale tool calls and processes."""
        from datetime import datetime, timedelta
        import dateutil.parser
        
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        # Clean up stale tool calls
        stale_tool_calls = []
        for tool_id, info in self._active_tool_calls.items():
            start_time = dateutil.parser.parse(info["start_time"])
            if start_time < cutoff_time:
                stale_tool_calls.append(tool_id)
        
        for tool_id in stale_tool_calls:
            await self.error_tool_call(tool_id, "Tool call timed out")
        
        # Clean up stale processes
        stale_processes = []
        for process_id, info in self._active_processes.items():
            start_time = dateutil.parser.parse(info["start_time"])
            if start_time < cutoff_time:
                stale_processes.append(process_id)
        
        for process_id in stale_processes:
            await self.error_process(process_id, "Process timed out")


# Global message service instance
_message_service = MessageService()


def get_message_service() -> MessageService:
    """Get the global message service instance."""
    return _message_service
