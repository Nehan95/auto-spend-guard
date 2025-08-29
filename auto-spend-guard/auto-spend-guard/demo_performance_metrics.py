#!/usr/bin/env python3
"""
Demonstration script for enhanced performance metrics with tool usage information.
This shows how the updated workflow tracks and displays comprehensive tool performance data.
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def demo_performance_metrics():
    """Demonstrate the enhanced performance metrics with tool usage tracking"""
    
    print("🚀 Enhanced Performance Metrics Demo with Tool Usage Tracking")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = SpendAnalyzerWorkflow()
    
    # Test with a specific question to show detailed metrics
    test_question = "What are our AWS costs for this month?"
    
    print(f"\n🧪 Testing Question: {test_question}")
    print("=" * 60)
    
    try:
        # Run the workflow
        result = workflow.run(test_question)
        
        print(f"\n✅ Workflow Completed Successfully!")
        print(f"📊 Classification: {result.get('classification', 'Unknown')}")
        print(f"🔧 Tools Used: {', '.join(result.get('tools_used', []))}")
        
        # Display comprehensive performance metrics
        print(f"\n{'='*60}")
        print("📈 COMPREHENSIVE PERFORMANCE METRICS")
        print(f"{'='*60}")
        
        workflow.display_latency_metrics(result)
        
        # Show specific tool performance highlights
        print(f"\n{'='*60}")
        print("🎯 TOOL PERFORMANCE HIGHLIGHTS")
        print(f"{'='*60}")
        
        tools_used = result.get('tools_used', [])
        relevant_data = result.get('relevant_data', {})
        latency_metrics = result.get('latency_metrics', {})
        
        if tools_used:
            print(f"🔧 Tools Executed: {len(tools_used)}")
            for tool in tools_used:
                print(f"   • {tool}")
            
            # Tool-specific metrics
            if "data_retrieval" in latency_metrics:
                dr_metrics = latency_metrics["data_retrieval"]
                tool_time = dr_metrics.get('tool_duration_seconds', 0)
                total_time = dr_metrics.get('total_duration_seconds', 0)
                
                print(f"\n⚡ Tool Performance:")
                print(f"   • Tool Execution Time: {tool_time:.3f}s")
                print(f"   • Total Retrieval Time: {total_time:.3f}s")
                
                if total_time > 0:
                    efficiency = (tool_time / total_time) * 100
                    print(f"   • Tool Efficiency: {efficiency:.1f}%")
                
                # Show tool details
                if "tool_used" in relevant_data:
                    print(f"   • Primary Tool: {relevant_data['tool_used']}")
                    print(f"   • Tool Description: {relevant_data.get('tool_description', 'N/A')}")
        
        # Show message state summary
        messages = result.get('messages', [])
        if messages:
            print(f"\n💬 Message State Summary:")
            print(f"   • Total Messages: {len(messages)}")
            
            # Count by type
            message_types = {}
            for msg in messages:
                msg_type = msg.__class__.__name__
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            for msg_type, count in message_types.items():
                print(f"   • {msg_type}: {count}")
        
        # Show cost analysis
        print(f"\n💰 Cost Analysis:")
        total_tokens = 0
        for step, metrics in latency_metrics.items():
            if isinstance(metrics, dict):
                step_tokens = metrics.get('total_tokens', 0)
                total_tokens += step_tokens
        
        estimated_cost = total_tokens * 0.00001  # GPT-4o-mini pricing
        print(f"   • Total Tokens: {total_tokens}")
        print(f"   • Estimated Cost: ${estimated_cost:.4f}")
        
        if tools_used:
            cost_per_tool = estimated_cost / len(tools_used)
            print(f"   • Cost per Tool: ${cost_per_tool:.4f}")
        
        print(f"\n{'='*60}")
        print("🎉 Performance Metrics Demo Complete!")
        print(f"{'='*60}")
        
        print("\nKey Features Demonstrated:")
        print("✅ Tool usage tracking and recording")
        print("✅ Tool-specific performance metrics")
        print("✅ Efficiency calculations and ratings")
        print("✅ Cost analysis per tool operation")
        print("✅ Message state tracking")
        print("✅ Comprehensive workflow analytics")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_performance_metrics()
