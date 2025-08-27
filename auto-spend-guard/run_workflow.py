#!/usr/bin/env python3
"""
Main script to run the Spend Analyzer LangGraph workflow
"""

import os
from dotenv import load_dotenv
from langgraph_workflow import SpendAnalyzerWorkflow
import json

def print_separator():
    """Print a separator line"""
    print("=" * 80)

def print_result(result):
    """Pretty print the workflow result"""
    print_separator()
    print("WORKFLOW EXECUTION RESULT")
    print_separator()
    
    print(f"Question: {result.get('question', 'N/A')}")
    print(f"Classification: {result.get('classification', 'N/A')}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return
    
    print(f"Final Answer:")
    print("-" * 40)
    print(result.get('final_answer', 'No answer generated'))
    
    # Print data summary
    relevant_data = result.get('relevant_data', {})
    if relevant_data and 'error' not in relevant_data:
        print("\nData Summary:")
        print("-" * 40)
        for key, value in relevant_data.items():
            if key not in ['sample_data', 'data_shape']:
                if isinstance(value, (list, tuple)):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {value}")

def main():
    """Main function to run the workflow"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    print("🚀 Initializing Spend Analyzer Workflow...")
    
    try:
        # Initialize the workflow
        workflow = SpendAnalyzerWorkflow()
        print("✅ Workflow initialized successfully!")
        
        # Display workflow information
        workflow.display_workflow_info()
        
        # Create workflow visualization (optional)
        print("\n🎨 Creating workflow visualization...")
        try:
            workflow.visualize_workflow("workflow_visualization.html")
        except Exception as e:
            print(f"⚠️  Visualization creation skipped: {str(e)}")
        
        print(f"\n📊 Loaded {len(workflow.data_loader.get_available_datasets())} datasets:")
        for dataset in workflow.data_loader.get_available_datasets():
            info = workflow.data_loader.get_dataframe_info(dataset)
            print(f"   • {dataset}: {info['shape'][0]} rows, {info['shape'][1]} columns")
        
        print("\n🎯 INTERACTIVE MODE")
        print("Ask your own questions (type 'quit' to exit):")
        
        while True:
            try:
                user_question = input("\n❓ Your question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_question:
                    continue
                
                print(f"\n🔄 Processing: {user_question}")
                result = workflow.run(user_question)
                print_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if all required packages are installed: pip install -r requirements.txt")
        print("2. Verify your OpenAI API key is set in .env file")
        print("3. Ensure the docs/ folder contains the required data files")

if __name__ == "__main__":
    main()
