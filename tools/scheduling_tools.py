"""
Simple scheduling tool for LLM to trigger UI popup.
"""

from typing import Dict, Any
from loguru import logger

from messaging.protocol import create_ui_update
from services.message_service import get_message_service
from pipecat.services.llm_service import FunctionCallParams


class SchedulingTools:
    """Simple scheduling tool - just opens popup"""
    
    def __init__(self):
        self.message_service = get_message_service()
    
    async def show_scheduling_popup(self, service_type: str = "consultation", context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Opens scheduling popup UI for user to fill out.
        
        Args:
            service_type: Type of service (consultation, demo, onboarding, support, training)
            context: Brief context (optional)
            **kwargs: Additional arguments from LLM service (ignored)
        """
        try:
            # Send UI update to show popup
            await self.message_service.send_message(
                create_ui_update(
                    component_id="scheduling_popup",
                    action="show_popup",
                    data={
                        "service_type": service_type,
                        "context": context,
                        "business_hours": "9 AM - 5 PM (Mon-Fri)",
                        "timezone_default": "UTC"
                    }
                )
            )
            
            logger.info(f"Triggered scheduling popup for {service_type}")
            
            return {
                "success": True,
                "message": f"Opening scheduling form for {service_type}",
                "popup_triggered": True
            }
            
        except Exception as e:
            logger.error(f"Error showing scheduling popup: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Direct function for LLM registration using new FunctionCallParams
async def show_scheduling_popup(params: FunctionCallParams, **kwargs):
    """
    Opens a scheduling popup when user wants to book an appointment.
    
    Args:
        params: Function call parameters from LLM service
        **kwargs: Additional parameters including service_type and context
    """
    try:
        # Extract parameters
        service_type = kwargs.get("service_type", "consultation")
        context = kwargs.get("context", "")
        
        # Get the tools instance and call the method
        tools = get_scheduling_tools()
        result = await tools.show_scheduling_popup(service_type, context)
        
        # Return result via callback
        await params.result_callback(result)
        
    except Exception as e:
        logger.error(f"Error in show_scheduling_popup: {e}")
        await params.result_callback({
            "success": False,
            "error": str(e)
        })


# Global instance
_scheduling_tools = None

def get_scheduling_tools() -> SchedulingTools:
    """Get the global scheduling tools instance"""
    global _scheduling_tools
    if _scheduling_tools is None:
        _scheduling_tools = SchedulingTools()
    return _scheduling_tools
