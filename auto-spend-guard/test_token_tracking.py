#!/usr/bin/env python3
"""
Test script for Token Tracking in LangGraph Workflow
Demonstrates token counting and cost estimation capabilities
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def test_token_tracking():
    """Test the token tracking functionality"""
    print("🧪 TESTING TOKEN TRACKING IN LANGRAPH WORKFLOW")
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
        print("🚀 Running workflow with token tracking...")
        result = workflow.run({
            "question": test_question
        })
        
        print("\n✅ Workflow completed successfully!")
        
        # Display token metrics
        if "latency_metrics" in result:
            print("\n🔤 TOKEN USAGE BREAKDOWN:")
            print("=" * 40)
            
            # Question Classification Tokens
            qc_metrics = result["latency_metrics"].get("question_classification", {})
            if qc_metrics:
                print(f"\n🔍 Question Classification:")
                print(f"   • Input Tokens: {qc_metrics.get('input_tokens', 0)}")
                print(f"   • Output Tokens: {qc_metrics.get('output_tokens', 0)}")
                print(f"   • Total Tokens: {qc_metrics.get('total_tokens', 0)}")
            
            # Response Generation Tokens
            rg_metrics = result["latency_metrics"].get("response_generation", {})
            if rg_metrics:
                print(f"\n🔍 Response Generation:")
                print(f"   • Input Tokens: {rg_metrics.get('input_tokens', 0)}")
                print(f"   • Output Tokens: {rg_metrics.get('output_tokens', 0)}")
                print(f"   • Total Tokens: {rg_metrics.get('total_tokens', 0)}")
            
            # Calculate total tokens and cost
            total_input = sum([
                qc_metrics.get('input_tokens', 0),
                rg_metrics.get('input_tokens', 0)
            ])
            total_output = sum([
                qc_metrics.get('output_tokens', 0),
                rg_metrics.get('output_tokens', 0)
            ])
            total_tokens = total_input + total_output
            
            print(f"\n📊 TOKEN SUMMARY:")
            print(f"   • Total Input Tokens: {total_input}")
            print(f"   • Total Output Tokens: {total_output}")
            print(f"   • Total Tokens: {total_tokens}")
            
            # Cost estimation for different models
            print(f"\n💰 COST ESTIMATION:")
            print(f"   • GPT-3.5-turbo: ${(total_tokens * 0.000002):.4f}")
            print(f"   • GPT-4: ${(total_tokens * 0.00003):.4f}")
            print(f"   • GPT-4-turbo: ${(total_tokens * 0.00001):.4f}")
            
        else:
            print("\n❌ No latency metrics found in result")
        
        # Display full latency metrics if available
        if hasattr(workflow, 'display_latency_metrics'):
            print("\n" + "="*60)
            print("📊 FULL LATENCY & TOKEN METRICS")
            print("="*60)
            workflow.display_latency_metrics(result)
        
        print("\n" + "="*60)
        print("🎉 Token tracking test completed!")
        print("\n🚀 Token Tracking Features:")
        print("   • Real-time token counting for all LLM operations")
        print("   • Input vs. output token breakdown")
        print("   • Cost estimation for multiple GPT models")
        print("   • Performance optimization insights")
        print("   • Budget planning and cost control")
        
    except Exception as e:
        print(f"❌ Error testing token tracking: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_token_tracking()
