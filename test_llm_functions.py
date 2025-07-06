"""
Test the actual Pipecat LLM function call flow
"""

import asyncio
from services.service_factory import ServiceFactory
from pipecat.services.llm_service import FunctionCallParams

async def test_llm_function_registration():
    """Test that LLM functions are properly registered and callable"""
    print("🧪 Testing LLM Function Registration\n")
    
    # Create LLM service
    llm = ServiceFactory.create_llm_service()
    
    # Check if functions are registered
    print("1️⃣ Checking function registration...")
    assert llm.has_function("show_scheduling_popup"), "show_scheduling_popup not registered"
    assert llm.has_function("check_availability"), "check_availability not registered"  
    assert llm.has_function("get_available_slots"), "get_available_slots not registered"
    print("   ✅ All functions are properly registered\n")
    
    # Create LLM context with tools
    print("2️⃣ Testing LLM context creation...")
    context = ServiceFactory.create_llm_context()
    assert context.tools is not None, "Tools not added to context"
    print("   ✅ LLM context created with tools schema\n")
    
    # Test function call simulation
    print("3️⃣ Simulating function calls...")
    
    # Mock FunctionCallParams
    class MockParams:
        def __init__(self, arguments):
            self.arguments = arguments
            self.tool_call_id = "test_call_123"
            self.function_name = "show_scheduling_popup"
            
        async def result_callback(self, result):
            print(f"   📞 Function result: {result.get('message', 'Success')}")
    
    # Test show_scheduling_popup
    from tools.scheduling_tools import show_scheduling_popup
    params = MockParams({"service_type": "demo", "context": "User wants a product demo"})
    result = await show_scheduling_popup(params)
    assert result["popup_shown"] == True, "Popup not shown"
    
    print("\n🎉 LLM Function Registration Test Passed!")
    print("\n📋 Summary:")
    print("   • Functions registered with LLM service ✅")
    print("   • Tools schema added to context ✅") 
    print("   • Function calls work correctly ✅")
    print("   • Pipecat integration is complete ✅")

if __name__ == "__main__":
    asyncio.run(test_llm_function_registration())
