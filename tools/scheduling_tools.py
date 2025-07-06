"""
Scheduling tool calls for LLM to use following protocol.py standards.
"""

import asyncio
import uuid
from typing import Dict, Any, List
from loguru import logger

from messaging.protocol import (
    create_tool_call_start, create_tool_call_progress, create_tool_call_result, create_tool_call_error
)
from services.scheduling_service import get_scheduling_service, SchedulingRequest
from services.message_service import get_message_service


class SchedulingTools:
    """Tool calls for scheduling functionality"""
    
    def __init__(self):
        self.scheduling_service = get_scheduling_service()
        self.message_service = get_message_service()
    
    async def show_scheduling_popup(self, service_type: str = "consultation", context: str = "") -> Dict[str, Any]:
        """
        Tool call to show scheduling popup UI to user.
        This is the ONLY tool the LLM should use for scheduling.
        DO NOT collect date, time, or email through conversation.
        
        Args:
            service_type: Type of service to schedule (consultation, demo, onboarding, support, training)
            context: Brief context about why this is being scheduled (optional)
        """
        tool_id = f"schedule_popup_{uuid.uuid4().hex[:8]}"
        
        try:
            # Send tool call start message
            await self.message_service.send_message(
                create_tool_call_start("show_scheduling_popup", tool_id, {
                    "service_type": service_type,
                    "context": context
                })
            )
            
            # Send progress message
            await self.message_service.send_message(
                create_tool_call_progress(tool_id, {
                    "status": "displaying_popup",
                    "message": "Opening scheduling interface..."
                })
            )
            
            # Send UI update to show scheduling popup
            from messaging.protocol import create_ui_update
            await self.message_service.send_message(
                create_ui_update(
                    component_id="scheduling_popup",
                    action="show_popup",
                    data={
                        "service_type": service_type,
                        "context": context,
                        "available_services": [
                            "consultation", "demo", "onboarding", "support", "training"
                        ],
                        "business_hours": "9 AM - 5 PM (Mon-Fri)",
                        "timezone": "UTC"
                    }
                )
            )
            
            # Send tool call result
            result = {
                "popup_shown": True,
                "service_type": service_type,
                "tool_id": tool_id,
                "message": f"Scheduling popup displayed for {service_type}. User can now select date, time, and email."
            }
            
            await self.message_service.send_message(
                create_tool_call_result(tool_id, result, "show_scheduling_popup")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error showing scheduling popup: {e}")
            await self.message_service.send_message(
                create_tool_call_error(tool_id, str(e), "show_scheduling_popup")
            )
            return {"error": str(e), "popup_shown": False}
    
    async def check_availability(self, date: str, time: str, service_type: str = "consultation") -> Dict[str, Any]:
        """
        Tool call to check availability for a specific date/time.
        
        Args:
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format  
            service_type: Type of service
        """
        tool_id = f"check_availability_{uuid.uuid4().hex[:8]}"
        
        try:
            # Send tool call start
            await self.message_service.send_message(
                create_tool_call_start("check_availability", tool_id, {
                    "date": date,
                    "time": time,
                    "service_type": service_type
                })
            )
            
            # Create temporary request for validation
            temp_request = SchedulingRequest(
                email="temp@example.com",  # Temporary email for validation
                date=date,
                time=time,
                service_type=service_type
            )
            
            # Check availability using scheduling service
            availability = await self.scheduling_service._check_availability(temp_request)
            
            result = {
                "available": availability["available"],
                "date": date,
                "time": time,
                "service_type": service_type,
                "reason": availability.get("reason", "")
            }
            
            # Send tool call result
            await self.message_service.send_message(
                create_tool_call_result(tool_id, result, "check_availability")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            await self.message_service.send_message(
                create_tool_call_error(tool_id, str(e), "check_availability")
            )
            return {"error": str(e), "available": False}
    
    async def get_available_slots(self, date: str, service_type: str = "consultation") -> Dict[str, Any]:
        """
        Tool call to get available time slots for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            service_type: Type of service
        """
        tool_id = f"get_slots_{uuid.uuid4().hex[:8]}"
        
        try:
            # Send tool call start
            await self.message_service.send_message(
                create_tool_call_start("get_available_slots", tool_id, {
                    "date": date,
                    "service_type": service_type
                })
            )
            
            # Generate available slots (mock implementation)
            # In real implementation, this would query a calendar system
            business_hours = list(range(9, 17))  # 9 AM to 5 PM
            available_slots = []
            
            for hour in business_hours:
                for minute in [0, 30]:  # 30-minute intervals
                    time_slot = f"{hour:02d}:{minute:02d}"
                    # Check if this slot is available
                    availability = await self.check_availability(date, time_slot, service_type)
                    if availability.get("available", False):
                        available_slots.append(time_slot)
            
            result = {
                "date": date,
                "service_type": service_type,
                "available_slots": available_slots,
                "total_slots": len(available_slots)
            }
            
            # Send UI update with available slots
            from messaging.protocol import create_ui_update
            await self.message_service.send_message(
                create_ui_update(
                    component_id="scheduling_popup",
                    action="update_available_slots",
                    data=result
                )
            )
            
            # Send tool call result
            await self.message_service.send_message(
                create_tool_call_result(tool_id, result, "get_available_slots")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting available slots: {e}")
            await self.message_service.send_message(
                create_tool_call_error(tool_id, str(e), "get_available_slots")
            )
            return {"error": str(e), "available_slots": []}


# Global instance
_scheduling_tools = None

def get_scheduling_tools() -> SchedulingTools:
    """Get the global scheduling tools instance"""
    global _scheduling_tools
    if _scheduling_tools is None:
        _scheduling_tools = SchedulingTools()
    return _scheduling_tools


# Tool function definitions for LLM
async def show_scheduling_popup(params) -> Dict[str, Any]:
    """
    Show scheduling popup for appointment booking.
    This is the main scheduling tool - use this when user wants to schedule anything.
    The popup will handle all date/time/email collection.
    """
    tools = get_scheduling_tools()
    
    # Extract arguments from FunctionCallParams
    arguments = params.arguments if hasattr(params, 'arguments') else params
    service_type = arguments.get("service_type", "consultation")
    context = arguments.get("context", "")
    
    result = await tools.show_scheduling_popup(service_type, context)
    
    # Call the result callback if available
    if hasattr(params, 'result_callback'):
        await params.result_callback(result)
    
    return result


async def check_availability(params) -> Dict[str, Any]:
    """Check if a specific date/time slot is available"""
    tools = get_scheduling_tools()
    
    # Extract arguments
    arguments = params.arguments if hasattr(params, 'arguments') else params
    date = arguments.get("date", "")
    time = arguments.get("time", "")
    service_type = arguments.get("service_type", "consultation")
    
    result = await tools.check_availability(date, time, service_type)
    
    # Call the result callback if available
    if hasattr(params, 'result_callback'):
        await params.result_callback(result)
    
    return result


async def get_available_slots(params) -> Dict[str, Any]:
    """Get all available time slots for a specific date"""
    tools = get_scheduling_tools()
    
    # Extract arguments
    arguments = params.arguments if hasattr(params, 'arguments') else params
    date = arguments.get("date", "")
    service_type = arguments.get("service_type", "consultation")
    
    result = await tools.get_available_slots(date, service_type)
    
    # Call the result callback if available
    if hasattr(params, 'result_callback'):
        await params.result_callback(result)
    
    return result
