"""
Example usage of the standardized message protocol for complex multi-step operations.
"""

import asyncio
import json
from typing import List, Dict, Any
from services.message_service import get_message_service
from messaging.protocol import (
    create_scheduler_flow_messages, create_booking_flow_messages,
    create_tool_call_start, create_process_start
)


class ExampleWorkflows:
    """Example workflows demonstrating the message protocol."""
    
    def __init__(self):
        self.message_service = get_message_service()
    
    async def complex_scheduling_workflow(self, tasks: List[Dict[str, Any]]):
        """Example: Complex scheduling workflow with multiple steps."""
        
        # Start the overall process
        process_id = await self.message_service.start_process(
            name="Complex Scheduling",
            total_steps=len(tasks) + 2,  # +2 for validation and finalization
            description="Processing multiple scheduling requests with validation"
        )
        
        try:
            # Step 1: Validate all requests
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Validating Requests",
                status="running",
                description="Checking for conflicts and availability"
            )
            
            # Simulate validation
            await asyncio.sleep(1)
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Validating Requests",
                status="completed",
                data={"validated_count": len(tasks)}
            )
            
            # Step 2: Process each task
            for i, task in enumerate(tasks):
                tool_id = await self.message_service.start_tool_call(
                    tool_name="calendar_scheduler",
                    parameters=task
                )
                
                await self.message_service.update_process_step(
                    process_id=process_id,
                    step_name=f"Scheduling {task.get('name', f'Task {i+1}')}",
                    status="running",
                    data={"tool_call_id": tool_id, "task": task}
                )
                
                # Simulate tool progress
                await self.message_service.update_tool_call(
                    tool_id=tool_id,
                    progress={"status": "checking_availability", "progress": 25}
                )
                
                await asyncio.sleep(0.5)
                
                await self.message_service.update_tool_call(
                    tool_id=tool_id,
                    progress={"status": "booking_slot", "progress": 75}
                )
                
                await asyncio.sleep(0.5)
                
                # Complete the tool call
                await self.message_service.complete_tool_call(
                    tool_id=tool_id,
                    result={"scheduled_time": task.get('time'), "confirmation_id": f"conf_{i}"}
                )
                
                await self.message_service.update_process_step(
                    process_id=process_id,
                    step_name=f"Scheduling {task.get('name', f'Task {i+1}')}",
                    status="completed"
                )
            
            # Step 3: Finalization
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Finalizing Schedule",
                status="running",
                description="Sending confirmations and updating calendar"
            )
            
            await asyncio.sleep(1)
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Finalizing Schedule",
                status="completed"
            )
            
            # Complete the entire process
            await self.message_service.complete_process(
                process_id=process_id,
                result={
                    "total_scheduled": len(tasks),
                    "completion_time": "2024-01-15T10:30:00Z"
                }
            )
            
        except Exception as e:
            await self.message_service.error_process(process_id, str(e))
    
    async def booking_workflow_with_payment(self, booking_details: Dict[str, Any]):
        """Example: Booking workflow with payment processing."""
        
        # Start booking process
        process_id = await self.message_service.start_process(
            name="Hotel Booking",
            total_steps=4,
            description=f"Booking {booking_details.get('room_type')} for {booking_details.get('guest_name')}"
        )
        
        try:
            # Step 1: Check availability
            availability_tool = await self.message_service.start_tool_call(
                tool_name="availability_checker",
                parameters={"dates": booking_details.get("dates"), "room_type": booking_details.get("room_type")}
            )
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Checking Availability",
                status="running"
            )
            
            await asyncio.sleep(2)
            
            await self.message_service.complete_tool_call(
                availability_tool,
                {"available": True, "price": 150.00}
            )
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Checking Availability",
                status="completed",
                data={"available": True, "price": 150.00}
            )
            
            # Step 2: Process payment
            payment_tool = await self.message_service.start_tool_call(
                tool_name="payment_processor",
                parameters={"amount": 150.00, "card": booking_details.get("payment")}
            )
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Processing Payment",
                status="running"
            )
            
            await self.message_service.update_tool_call(
                payment_tool,
                {"status": "authorizing", "progress": 50}
            )
            
            await asyncio.sleep(1.5)
            
            await self.message_service.complete_tool_call(
                payment_tool,
                {"transaction_id": "txn_12345", "status": "completed"}
            )
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Processing Payment",
                status="completed"
            )
            
            # Step 3: Confirm booking
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Confirming Booking",
                status="running"
            )
            
            await asyncio.sleep(1)
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Confirming Booking",
                status="completed"
            )
            
            # Step 4: Send confirmation
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Sending Confirmation",
                status="running"
            )
            
            await asyncio.sleep(0.5)
            
            await self.message_service.update_process_step(
                process_id=process_id,
                step_name="Sending Confirmation",
                status="completed"
            )
            
            # Complete process
            await self.message_service.complete_process(
                process_id=process_id,
                result={
                    "booking_id": "book_67890",
                    "confirmation_code": "CONF123",
                    "total_amount": 150.00
                }
            )
            
            # Send follow-up questions
            await self.message_service.send_follow_up_questions([
                "Would you like to add any special requests?",
                "Should I send the confirmation to your email?",
                "Do you need transportation to the hotel?"
            ])
            
        except Exception as e:
            await self.message_service.error_process(process_id, str(e))


async def demo_message_protocol():
    """Demonstrate the message protocol with examples."""
    
    workflows = ExampleWorkflows()
    
    print("=== Demonstrating Standard Message Protocol ===\n")
    
    # Example 1: Simple follow-up questions
    print("1. Sending follow-up questions...")
    await workflows.message_service.send_follow_up_questions([
        "What time works best for you?",
        "Should I set a reminder?",
        "Would you like to make this recurring?"
    ])
    
    # Example 2: Complex scheduling workflow
    print("2. Running complex scheduling workflow...")
    tasks = [
        {"name": "Team Meeting", "time": "10:00 AM", "duration": "1 hour"},
        {"name": "Client Call", "time": "2:00 PM", "duration": "30 minutes"},
        {"name": "Project Review", "time": "4:00 PM", "duration": "45 minutes"}
    ]
    
    await workflows.complex_scheduling_workflow(tasks)
    
    # Example 3: Booking workflow
    print("3. Running booking workflow...")
    booking_details = {
        "guest_name": "John Doe",
        "room_type": "Deluxe Suite",
        "dates": {"check_in": "2024-02-01", "check_out": "2024-02-03"},
        "payment": {"card_number": "****-****-****-1234"}
    }
    
    await workflows.booking_workflow_with_payment(booking_details)
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_message_protocol())
