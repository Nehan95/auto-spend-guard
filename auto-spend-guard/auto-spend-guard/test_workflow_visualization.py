#!/usr/bin/env python3
"""
Test script for the enhanced workflow visualization.
This demonstrates the new workflow structure with retriever agent and tool routing.
"""

from langgraph_workflow import SpendAnalyzerWorkflow
from workflow_visualizer import WorkflowVisualizer

def test_enhanced_workflow_visualization():
    """Test the enhanced workflow visualization with retriever agent"""
    
    print("🚀 Testing Enhanced Workflow Visualization with Retriever Agent")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = SpendAnalyzerWorkflow()
    
    # Display workflow information
    workflow.display_workflow_info()
    
    # Test the enhanced workflow visualization
    print(f"\n🎨 Testing Enhanced Workflow Visualization:")
    print("=" * 60)
    
    try:
        # Create enhanced workflow visualization
        html_file = workflow.visualize_workflow("enhanced_workflow_demo.html")
        
        if html_file:
            print(f"✅ Enhanced workflow visualization created successfully!")
            print(f"📁 File: {html_file}")
            print(f"🌐 Open this file in your browser to view the interactive workflow diagram")
            
            # Show file details
            import os
            if os.path.exists(html_file):
                file_size = os.path.getsize(html_file)
                print(f"📏 File size: {file_size:,} bytes")
                print(f"📅 Created: {os.path.getctime(html_file)}")
        else:
            print("❌ Failed to create enhanced workflow visualization")
            
    except Exception as e:
        print(f"❌ Error testing workflow visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test the visualizer directly
    print(f"\n🎨 Testing WorkflowVisualizer Directly:")
    print("=" * 60)
    
    try:
        visualizer = WorkflowVisualizer()
        
        # Test HTML creation
        html_file = visualizer.create_workflow_html("direct_visualizer_test.html")
        if html_file:
            print(f"✅ Direct visualizer test successful!")
            print(f"📁 File: {html_file}")
        else:
            print("❌ Direct visualizer test failed")
        
        # Test text display
        print(f"\n📊 Text-based workflow display:")
        visualizer.display_text_workflow()
        
    except Exception as e:
        print(f"❌ Error testing direct visualizer: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test workflow execution to see the new structure in action
    print(f"\n🧪 Testing Workflow Execution with New Structure:")
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
    
    print(f"\n🎯 Enhanced Workflow Visualization Testing Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Enhanced workflow visualization with retriever agent")
    print("✅ Interactive HTML workflow diagrams")
    print("✅ Text-based workflow structure display")
    print("✅ Workflow execution with new retriever agent")
    print("✅ Performance metrics and tool usage tracking")
    print("✅ Complete workflow state management")

if __name__ == "__main__":
    test_enhanced_workflow_visualization()
