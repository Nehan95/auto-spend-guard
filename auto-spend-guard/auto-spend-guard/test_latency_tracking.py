#!/usr/bin/env python3
"""
Test script for Latency Tracking in LangGraph Workflow
Demonstrates performance monitoring capabilities
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def test_latency_tracking():
    """Test the latency tracking functionality"""
    print("🧪 TESTING LATENCY TRACKING IN LANGRAPH WORKFLOW")
    print("=" * 60)
    
    try:
        # Initialize workflow
        workflow = SpendAnalyzerWorkflow()
        print("✅ Workflow initialized successfully!")
        
        # Test question to trigger the workflow
        test_question = "What are our AWS costs and which services are most expensive?"
        print(f"\n📝 Test Question: {test_question}")
        print("-" * 50)
        
        # Run the workflow
        print("🚀 Running workflow with latency tracking...")
        result = workflow.run({
            "question": test_question
        })
        
        print("\n✅ Workflow completed successfully!")
        
        # Display latency metrics
        if hasattr(workflow, 'display_latency_metrics'):
            workflow.display_latency_metrics(result)
        else:
            print("\n📊 Latency metrics available in result:")
            if "latency_metrics" in result:
                print(f"   • Question Classification: {result['latency_metrics'].get('question_classification', {}).get('total_duration_seconds', 0)}s")
                print(f"   • Data Retrieval: {result['latency_metrics'].get('data_retrieval', {}).get('total_duration_seconds', 0)}s")
                print(f"   • Response Generation: {result['latency_metrics'].get('response_generation', {}).get('total_duration_seconds', 0)}s")
            else:
                print("   • No latency metrics found in result")
        
        print("\n" + "="*60)
        print("🎉 Latency tracking test completed!")
        print("\n🚀 Performance Monitoring Features:")
        print("   • Real-time latency tracking for each workflow step")
        print("   • LLM vs. overhead timing breakdown")
        print("   • Agent performance metrics")
        print("   • Response generation timing")
        print("   • Comprehensive performance summary")
        
    except Exception as e:
        print(f"❌ Error testing latency tracking: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_latency_tracking()
