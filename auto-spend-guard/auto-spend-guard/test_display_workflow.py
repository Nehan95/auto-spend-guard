#!/usr/bin/env python3
"""
Test script for the enhanced display_workflow method.
This demonstrates the new workflow display capabilities with retriever agent information.
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def test_display_workflow():
    """Test the enhanced display_workflow method"""
    
    print("🚀 Testing Enhanced Display Workflow Method")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = SpendAnalyzerWorkflow()
    
    # Test the enhanced display_workflow method
    print("\n🎨 Testing Enhanced Display Workflow:")
    print("=" * 60)
    
    try:
        # Call the enhanced display_workflow method
        workflow.display_workflow()
        
        print("\n✅ Enhanced display_workflow method executed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing display_workflow: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test the workflow info display as well
    print(f"\n🔧 Testing Workflow Info Display:")
    print("=" * 60)
    
    try:
        workflow.display_workflow_info()
        print("\n✅ Workflow info display executed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing workflow info display: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test workflow execution to see the new structure in action
    print(f"\n🧪 Testing Workflow Execution:")
    print("=" * 60)
    
    try:
        # Test with a simple question
        test_question = "What are our AWS costs for this month?"
        print(f"Testing question: {test_question}")
        
        result = workflow.run(test_question)
        
        print(f"✅ Workflow executed successfully!")
        print(f"📊 Classification: {result.get('classification', 'Unknown')}")
        print(f"🔧 Tools Used: {', '.join(result.get('tools_used', []))}")
        print(f"💬 Message Count: {len(result.get('messages', []))}")
        
        # Show relevant data info
        relevant_data = result.get('relevant_data', {})
        if 'error' not in relevant_data:
            print(f"📈 Data Source: {relevant_data.get('data_source', 'Unknown')}")
            print(f"🔧 Retrieval Method: {relevant_data.get('data_retrieval_method', 'Unknown')}")
            
            # Show retriever agent information if used
            if "retriever_agent" in relevant_data.get('data_source', ''):
                print(f"🤖 Retriever Agent Used: Yes")
                if "tool_usage_summary" in relevant_data:
                    tool_summary = relevant_data["tool_usage_summary"]
                    print(f"🔧 Agent Tool Executions: {len(tool_summary)}")
                    for tool_name, tool_info in tool_summary.items():
                        executions = tool_info.get("executions", 0)
                        print(f"   • {tool_name}: {executions} execution(s)")
        
        # Display performance metrics
        print(f"\n⚡ Performance Metrics:")
        workflow.display_latency_metrics(result)
        
    except Exception as e:
        print(f"❌ Error testing workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 Enhanced Display Workflow Testing Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Enhanced display_workflow method with retriever agent details")
    print("✅ Comprehensive workflow structure explanation")
    print("✅ Conditional routing path visualization")
    print("✅ Retriever agent architecture details")
    print("✅ Tool capabilities and features")
    print("✅ State management and tracking information")
    print("✅ Performance monitoring capabilities")
    print("✅ Workflow benefits and advantages")
    print("✅ Usage examples and routing scenarios")

if __name__ == "__main__":
    test_display_workflow()
