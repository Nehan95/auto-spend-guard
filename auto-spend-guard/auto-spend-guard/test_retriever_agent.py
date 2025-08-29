#!/usr/bin/env python3
"""
Test script for the retriever agent functionality.
This demonstrates how the retriever agent intelligently routes classification responses to specific tools.
"""

from langgraph_workflow import SpendAnalyzerWorkflow
import json

def test_retriever_agent():
    """Test the retriever agent with different types of questions"""
    
    print("🚀 Testing Retriever Agent with Intelligent Tool Routing")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = SpendAnalyzerWorkflow()
    
    # Display workflow information
    workflow.display_workflow_info()
    
    # Test questions for different classifications
    test_questions = [
        "What are our AWS costs for this month?",
        "How is our budget tracking across projects?",
        "What are our vendor spending patterns?",
        "Show me our cloud infrastructure costs and budget variances",
        "Analyze our spending across AWS, vendors, and project budgets"
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
                print(f"\n📈 DATA RETRIEVAL DETAILS:")
                print(f"   • Data Source: {relevant_data.get('data_source', 'Unknown')}")
                print(f"   • Retrieval Method: {relevant_data.get('data_retrieval_method', 'Unknown')}")
                
                # Show retriever agent information
                if "retriever_agent" in relevant_data.get('data_source', ''):
                    print(f"   • Agent Data Available: Yes")
                    print(f"   • Tool Execution Steps: {len(relevant_data.get('tool_execution', []))}")
                    
                    # Show tool usage summary
                    if "tool_usage_summary" in relevant_data:
                        tool_summary = relevant_data["tool_usage_summary"]
                        print(f"   • Agent Tool Usage Summary:")
                        for tool_name, tool_info in tool_summary.items():
                            executions = tool_info.get("executions", 0)
                            print(f"     - {tool_name}: {executions} execution(s)")
                    
                    # Show agent data preview
                    agent_data = relevant_data.get('agent_data', '')
                    if agent_data:
                        print(f"   • Agent Response Preview:")
                        preview = agent_data[:300] + "..." if len(agent_data) > 300 else agent_data
                        print(f"     {preview}")
                
                # Show retrieval metadata
                metadata = relevant_data.get('retrieval_metadata', {})
                print(f"\n🔍 RETRIEVAL METADATA:")
                print(f"   • Response Length: {metadata.get('response_length', 0)} characters")
                print(f"   • Tools Executed: {metadata.get('tools_executed', 0)}")
                print(f"   • Retrieval Method: {metadata.get('retrieval_method', 'Unknown')}")
                
                if "agent_duration_seconds" in metadata:
                    print(f"   • Agent Duration: {metadata.get('agent_duration_seconds', 0):.3f}s")
                if "tool_duration_seconds" in metadata:
                    print(f"   • Tool Duration: {metadata.get('tool_duration_seconds', 0):.3f}s")
            else:
                print(f"❌ ERROR: {relevant_data['error']}")
            
            # Display final answer summary
            final_answer = result.get('final_answer', '')
            if final_answer:
                print(f"\n💡 FINAL ANSWER PREVIEW:")
                print(f"   {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
            
            # Display enhanced performance metrics
            print(f"\n🔧 ENHANCED PERFORMANCE METRICS:")
            workflow.display_latency_metrics(result)
            
        except Exception as e:
            print(f"❌ ERROR running workflow: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}")
    
    print("\n🎯 RETRIEVER AGENT TESTING COMPLETE!")
    print("\nKey Features Demonstrated:")
    print("✅ Intelligent tool routing based on classification")
    print("✅ Retriever agent with fallback to direct tool execution")
    print("✅ Tool usage tracking and execution summaries")
    print("✅ Enhanced performance metrics for agent operations")
    print("✅ Message state tracking throughout the workflow")
    print("✅ Comprehensive tool performance analytics")

def test_retriever_agent_direct():
    """Test the retriever agent directly to see its capabilities"""
    
    print("\n" + "="*80)
    print("🧪 DIRECT RETRIEVER AGENT TESTING")
    print("="*80)
    
    try:
        # Initialize the workflow to access the retriever agent
        workflow = SpendAnalyzerWorkflow()
        
        # Test the retriever agent directly
        test_cases = [
            ("What are our AWS costs?", "aws_costs"),
            ("Show budget tracking data", "budget"),
            ("Vendor spending analysis", "vendor_spend")
        ]
        
        for question, expected_classification in test_cases:
            print(f"\n🔍 Testing: {question}")
            print(f"Expected Classification: {expected_classification}")
            
            # Test the retriever agent directly
            result = workflow.retriever_agent.retrieve_data(question, expected_classification)
            
            if result["success"]:
                print(f"✅ Agent Success: Yes")
                print(f"   • Data Length: {len(result['data'])} characters")
                print(f"   • Tool Execution Steps: {len(result.get('tool_execution', []))}")
                
                # Show tool usage summary
                tool_summary = workflow.retriever_agent.get_tool_usage_summary(
                    result.get('tool_execution', [])
                )
                print(f"   • Tools Used: {list(tool_summary.keys())}")
                
                # Show data preview
                data_preview = result['data'][:200] + "..." if len(result['data']) > 200 else result['data']
                print(f"   • Data Preview: {data_preview}")
            else:
                print(f"❌ Agent Failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error testing retriever agent directly: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the full workflow with retriever agent
    test_retriever_agent()
    
    # Test the retriever agent directly
    test_retriever_agent_direct()
