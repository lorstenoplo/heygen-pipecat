"""
Test script for scheduling integration
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_scheduling_tools():
    """Test the scheduling tools functionality"""
    try:
        from tools.scheduling_tools import get_scheduling_tools
        
        tools = get_scheduling_tools()
        
        print("Testing show_scheduling_popup...")
        result = await tools.show_scheduling_popup("demo", "Testing scheduling popup")
        print(f"Popup result: {result}")
        
        print("\nTesting check_availability...")
        result = await tools.check_availability("2025-07-15", "14:00", "consultation")
        print(f"Availability result: {result}")
        
        print("\nTesting get_available_slots...")
        result = await tools.get_available_slots("2025-07-15", "consultation")
        print(f"Available slots: {result}")
        
        print("\n✅ All tool tests passed!")
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_scheduling_service():
    """Test the scheduling service"""
    try:
        from services.scheduling_service import get_scheduling_service, SchedulingRequest
        
        service = get_scheduling_service()
        
        # Create a test request
        request = SchedulingRequest(
            email="test@example.com",
            date="2025-07-15",
            time="14:00",
            service_type="consultation",
            notes="This is a test booking"
        )
        
        print("Testing scheduling service...")
        result = await service.schedule_appointment(request)
        print(f"Scheduling result: {result}")
        
        if result.success:
            print("✅ Scheduling service test passed!")
        else:
            print(f"⚠️ Scheduling completed but with issues: {result.error_message}")
        
    except Exception as e:
        print(f"❌ Scheduling service test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_rtvi_action():
    """Test the RTVI action handler"""
    try:
        from actions.rtvi_actions import handle_schedule_appointment
        
        # Test arguments
        args = {
            "email": "test@example.com",
            "date": "2025-07-15", 
            "time": "14:00",
            "service_type": "consultation",
            "duration_minutes": 60,
            "timezone": "UTC",
            "notes": "Test booking via RTVI action"
        }
        
        print("Testing RTVI action handler...")
        result = await handle_schedule_appointment(None, None, args)
        print(f"RTVI action result: {result}")
        
        if result.get("success"):
            print("✅ RTVI action test passed!")
        else:
            print(f"⚠️ RTVI action completed but with issues: {result.get('error')}")
        
    except Exception as e:
        print(f"❌ RTVI action test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    print("🧪 Testing Scheduling Integration\n")
    print("="*50)
    
    await test_scheduling_tools()
    print("\n" + "="*50)
    
    await test_scheduling_service()
    print("\n" + "="*50)
    
    await test_rtvi_action()
    print("\n" + "="*50)
    
    print("\n🎉 All tests completed!")
    
    # Check if RESEND_API_KEY is set
    if not os.getenv("RESEND_API_KEY"):
        print("\n⚠️ Note: RESEND_API_KEY not set - email functionality will be disabled")
    else:
        print("\n✅ RESEND_API_KEY is configured")

if __name__ == "__main__":
    asyncio.run(main())
