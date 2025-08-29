#!/usr/bin/env python3
"""
Test script for the updated LangGraph workflow with message state and tool tracking.
This demonstrates how the workflow now uses classification to route to specific tools.
"""

from langgraph_workflow import SpendAnalyzerWorkflow
import json

def test_updated_workflow():
    """Test the updated workflow with different types of questions"""
    
    print("🚀 Testing Updated LangGraph Workflow with Message State and Tool Tracking")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = SpendAnalyzerWorkflow()
    
    # Display workflow information
    workflow.display_workflow_info()
    
    # Test questions for different classifications
    test_questions = [
        "What are our AWS costs for this month?",
        "How is our budget tracking across projects?",
        "What are our vendor spending patterns?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"🧪 TEST {i}: {question}")
        print(f"{'='*60}")
        
        try:
            # Run the workflow
            result = workflow.run(question)
            
            # Display results
            print(f"\n📊 CLASSIFICATION: {result.get('classification', 'Unknown')}")
            print(f"🔧 TOOLS USED: {', '.join(result.get('tools_used', []))}")
            print(f"💬 MESSAGE COUNT: {len(result.get('messages', []))}")
            
            # Display relevant data info
            relevant_data = result.get('relevant_data', {})
            if 'error' not in relevant_data:
                print(f"📈 DATA SOURCE: {relevant_data.get('data_source', 'Unknown')}")
                print(f"🔧 TOOL USED: {relevant_data.get('tool_used', 'Unknown')}")
                print(f"📏 RESPONSE LENGTH: {relevant_data.get('retrieval_metadata', {}).get('response_length', 0)} characters")
            else:
                print(f"❌ ERROR: {relevant_data['error']}")
            
            # Display final answer summary
            final_answer = result.get('final_answer', '')
            if final_answer:
                print(f"\n💡 FINAL ANSWER PREVIEW:")
                print(f"   {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
            
            # Display latency metrics
            print(f"\n⚡ LATENCY METRICS:")
            latency_metrics = result.get('latency_metrics', {})
            for step, metrics in latency_metrics.items():
                if isinstance(metrics, dict):
                    duration = metrics.get('total_duration_seconds', 0)
                    status = metrics.get('status', 'unknown')
                    print(f"   • {step}: {duration:.3f}s ({status})")
            
            # Display enhanced tool usage and performance metrics
            print(f"\n🔧 ENHANCED TOOL & PERFORMANCE METRICS:")
            workflow.display_latency_metrics(result)
            
            # Display message state summary
            messages = result.get('messages', [])
            if messages:
                print(f"\n💬 MESSAGE STATE SUMMARY:")
                message_types = {}
                for msg in messages:
                    msg_type = msg.__class__.__name__
                    message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                for msg_type, count in message_types.items():
                    print(f"   • {msg_type}: {count} messages")
                
                # Show workflow progression
                workflow_steps = [msg.content for msg in messages if 'initiated' in msg.content.lower()]
                if workflow_steps:
                    print(f"   • Workflow Steps: {len(workflow_steps)}")
            
        except Exception as e:
            print(f"❌ ERROR running workflow: {str(e)}")
        
        print(f"\n{'='*60}")
    
    print("\n🎯 WORKFLOW TESTING COMPLETE!")
    print("\nKey Improvements Demonstrated:")
    print("✅ Message state tracking throughout workflow")
    print("✅ Tool usage tracking and recording")
    print("✅ Conditional routing based on classification")
    print("✅ Direct tool execution instead of agent-based retrieval")
    print("✅ Comprehensive performance metrics")
    print("✅ Workflow history and progression tracking")

if __name__ == "__main__":
    test_updated_workflow()
