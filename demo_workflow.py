#!/usr/bin/env python3
"""
Demo script to show the workflow structure and data loading capabilities
without requiring OpenAI API key
"""

from dataloader import DataLoader
import json

def demo_data_loading():
    """Demonstrate data loading capabilities"""
    print("🚀 DEMO: Data Loading Capabilities")
    print("=" * 60)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load all data
    print("📊 Loading data files...")
    data = loader.load_all_data()
    
    print(f"\n✅ Successfully loaded {len(data)} datasets:")
    for name, df in data.items():
        print(f"   • {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return loader

def demo_data_exploration(loader):
    """Demonstrate data exploration capabilities"""
    print("\n🔍 DEMO: Data Exploration")
    print("=" * 60)
    
    datasets = loader.get_available_datasets()
    
    for dataset_name in datasets:
        print(f"\n📈 Dataset: {dataset_name}")
        print("-" * 40)
        
        df = loader.get_dataframe(dataset_name)
        info = loader.get_dataframe_info(dataset_name)
        
        print(f"Shape: {info['shape']}")
        print(f"Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
        
        # Show sample data
        print("Sample data:")
        for i, row in enumerate(info['sample_data'][:2]):
            print(f"  Row {i+1}: {dict(list(row.items())[:3])}{'...' if len(row) > 3 else ''}")

def demo_search_capabilities(loader):
    """Demonstrate search capabilities"""
    print("\n🔎 DEMO: Search Capabilities")
    print("=" * 60)
    
    # Search for AWS-related data
    print("Searching for 'AWS' across all datasets...")
    aws_results = loader.search_data("AWS")
    
    for dataset, result in aws_results.items():
        print(f"\n📊 {dataset}:")
        print(f"   Found {result['count']} matches")
        for i, match in enumerate(result['matches'][:2]):
            print(f"   Match {i+1}: {dict(list(match.items())[:3])}{'...' if len(match) > 3 else ''}")
    
    # Search for budget-related data
    print("\nSearching for 'budget' across all datasets...")
    budget_results = loader.search_data("budget")
    
    for dataset, result in budget_results.items():
        print(f"\n📊 {dataset}:")
        print(f"   Found {result['count']} matches")
        for i, match in enumerate(result['matches'][:2]):
            print(f"   Match {i+1}: {dict(list(match.items())[:3])}{'...' if len(match) > 3 else ''}")

def demo_workflow_structure():
    """Demonstrate the workflow structure"""
    print("\n🔄 DEMO: Workflow Structure")
    print("=" * 60)
    
    print("The LangGraph workflow consists of 3 main nodes:")
    print("\n1. 📝 CLASSIFY_QUESTION")
    print("   • Uses OpenAI to classify questions into categories:")
    print("     - aws_costs: AWS cloud infrastructure costs")
    print("     - budget: Project and team budget tracking")
    print("     - vendor_spend: Vendor and supplier expenses")
    
    print("\n2. 🔍 RETRIEVE_DATA")
    print("   • Based on classification, retrieves relevant data:")
    print("     - AWS costs: Daily costs, service breakdowns, trends")
    print("     - Budget: Project budgets, variances, team spending")
    print("     - Vendor: Contract details, spending, risk assessment")
    
    print("\n3. 💬 GENERATE_ANSWER")
    print("   • Creates comprehensive answers using retrieved data")
    print("   • Provides insights and recommendations")
    print("   • Formats response professionally")
    
    print("\n🔄 Workflow Flow:")
    print("   Question → Classification → Data Retrieval → Answer Generation")

def demo_capabilities():
    """Show the system's capabilities"""
    print("\n💡 DEMO: System Capabilities")
    print("=" * 60)
    
    capabilities = {
        "aws_costs": [
            "Cloud infrastructure cost analysis",
            "Service-by-service spending breakdown",
            "Daily cost trend analysis",
            "Anomaly detection in cloud spending"
        ],
        "budget": [
            "Project budget tracking and variance analysis",
            "Team performance and spending patterns",
            "Budget utilization and optimization insights",
            "Anomaly detection in budget variances"
        ],
        "vendor_spend": [
            "Vendor contract and spending analysis",
            "Risk assessment and variance tracking",
            "Contract renewal and optimization insights",
            "Anomaly detection in vendor spending"
        ]
    }
    
    for category, caps in capabilities.items():
        print(f"\n📊 {category.upper().replace('_', ' ')}:")
        for i, cap in enumerate(caps, 1):
            print(f"   {i}. {cap}")

def main():
    """Run the demo"""
    print("🎯 AUTO-SPEND LANGGRAPH WORKFLOW DEMO")
    print("=" * 60)
    print("This demo shows the system's capabilities without requiring OpenAI API key")
    print("=" * 60)
    
    try:
        # Demo data loading
        loader = demo_data_loading()
        
        # Demo data exploration
        demo_data_exploration(loader)
        
        # Demo search capabilities
        demo_search_capabilities(loader)
        
        # Demo workflow structure
        demo_workflow_structure()
        
        # Demo system capabilities
        demo_capabilities()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📋 Next Steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run the full workflow: python run_workflow.py")
        print("3. Ask questions interactively!")
        
        print("\n💡 The system automatically:")
        print("   • Loads and analyzes your financial data")
        print("   • Classifies questions intelligently")
        print("   • Retrieves relevant information")
        print("   • Detects anomalies and patterns")
        print("   • Generates professional, structured answers")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please check that all data files are present in the docs/ folder")

if __name__ == "__main__":
    main()
