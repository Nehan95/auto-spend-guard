#!/usr/bin/env python3
"""
Display Workflow Information Script
Shows the compiled LangGraph workflow structure without requiring OpenAI API key
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def main():
    """Display workflow information"""
    print("🔧 AUTO-SPEND WORKFLOW STRUCTURE DISPLAY")
    print("=" * 60)
    print("This script shows the compiled workflow structure")
    print("No OpenAI API key required for this display")
    print("=" * 60)
    
    try:
        # Initialize the workflow (this will show the compiled structure)
        print("\n🚀 Initializing workflow...")
        workflow = SpendAnalyzerWorkflow()
        
        # Display detailed workflow information
        workflow.display_workflow_info()
        
        # Create visual workflow graph
        print("\n🎨 Creating workflow visualization...")
        workflow.visualize_workflow("auto_spend_workflow.html")
        
        # Additional workflow details
        print("\n" + "="*60)
        print("🔍 ADDITIONAL WORKFLOW DETAILS")
        print("="*60)
        
        # Show available methods
        print("📋 Available Methods:")
        methods = [method for method in dir(workflow) if not method.startswith('_')]
        for method in methods:
            if callable(getattr(workflow, method)):
                print(f"   • {method}()")
        
        # Show data loader info
        print(f"\n📊 Data Loader Status:")
        print(f"   • Available datasets: {len(workflow.data_loader.get_available_datasets())}")
        for dataset in workflow.data_loader.get_available_datasets():
            info = workflow.data_loader.get_dataframe_info(dataset)
            print(f"   • {dataset}: {info['shape'][0]} rows × {info['shape'][1]} columns")
        
        print("\n" + "="*60)
        print("✅ Workflow display completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error displaying workflow: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if all required packages are installed")
        print("2. Ensure the docs/ folder contains the required data files")
        print("3. Verify the langgraph_workflow.py file is correct")

if __name__ == "__main__":
    main()
