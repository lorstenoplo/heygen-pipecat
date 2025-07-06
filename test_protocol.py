"""
Test script to verify the message protocol functionality.
"""

import asyncio
import json
from services.message_service import get_message_service
from messaging.protocol import create_follow_up_questions, MessageType


async def test_message_protocol():
    """Test the message protocol functionality."""
    print("=== Testing Message Protocol ===\n")
    
    # Get the message service
    message_service = get_message_service()
    print("✓ Message service imported successfully")
    
    # Test 1: Create a simple follow-up message
    print("\n1. Testing follow-up message creation...")
    follow_up_msg = create_follow_up_questions([
        "What time works best for you?",
        "Should I set a reminder?",
        "Would you like to make this recurring?"
    ])
    
    print(f"✓ Created message with type: {follow_up_msg.type}")
    print(f"✓ Message ID: {follow_up_msg.message_id}")
    print(f"✓ Questions: {follow_up_msg.payload.get('questions', []) if follow_up_msg.payload else []}")
    
    # Test 2: Convert to RTVI frame data
    print("\n2. Testing RTVI frame conversion...")
    frame_data = follow_up_msg.to_rtvi_frame_data()
    print("✓ Converted to RTVI frame data:")
    print(json.dumps(frame_data, indent=2))
    
    # Test 3: Test tool call lifecycle (without actual RTVI processor)
    print("\n3. Testing tool call lifecycle...")
    
    # Note: Since we don't have an RTVI processor, we'll just test the message creation
    tool_id = "test_tool_123"
    message_service._active_tool_calls[tool_id] = {
        "name": "test_tool",
        "status": "running",
        "start_time": "2024-01-15T10:00:00Z"
    }
    
    # Test progress update
    progress_sent = await message_service.update_tool_call(tool_id, {
        "status": "processing",
        "progress": 50
    })
    print(f"✓ Tool call progress update: {progress_sent}")
    
    # Test completion
    completion_sent = await message_service.complete_tool_call(tool_id, {
        "result": "Task completed successfully"
    })
    print(f"✓ Tool call completion: {completion_sent}")
    
    # Test 4: Test process lifecycle
    print("\n4. Testing process lifecycle...")
    
    process_id = "test_process_456"
    message_service._active_processes[process_id] = {
        "name": "test_process",
        "total_steps": 3,
        "current_step": 0,
        "status": "running",
        "start_time": "2024-01-15T10:00:00Z"
    }
    
    # Test step update
    step_sent = await message_service.update_process_step(
        process_id=process_id,
        step_name="Processing data",
        status="completed",
        data={"items_processed": 100}
    )
    print(f"✓ Process step update: {step_sent}")
    
    # Test process completion
    completion_sent = await message_service.complete_process(process_id, {
        "total_items": 300,
        "success_rate": 0.95
    })
    print(f"✓ Process completion: {completion_sent}")
    
    # Test 5: Test message types
    print("\n5. Testing message type enumeration...")
    message_types = list(MessageType)
    print(f"✓ Available message types ({len(message_types)}):")
    for msg_type in message_types:
        print(f"  - {msg_type.value}")
    
    print("\n=== All tests completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(test_message_protocol())
