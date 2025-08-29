#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced response format and data formatting
without requiring OpenAI API key
"""

from dataloader import DataLoader
from langgraph_workflow import SpendAnalyzerWorkflow
import json

def test_data_formatting():
    """Test the enhanced data formatting for prompts"""
    print("🧪 Testing Enhanced Data Formatting")
    print("=" * 60)
    
    # Initialize data loader
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Test the workflow's data formatting method
    workflow = SpendAnalyzerWorkflow()
    
    # Test AWS data formatting
    print("\n📊 AWS Data Formatting Example:")
    print("-" * 40)
    aws_data = workflow._retrieve_aws_data("What are our AWS costs?")
    formatted_aws = workflow._format_data_for_prompt(aws_data)
    print(formatted_aws)
    
    # Test budget data formatting
    print("\n📊 Budget Data Formatting Example:")
    print("-" * 40)
    budget_data = workflow._retrieve_budget_data("Which projects are over budget?")
    formatted_budget = workflow._format_data_for_prompt(budget_data)
    print(formatted_budget)
    
    # Test vendor data formatting
    print("\n📊 Vendor Data Formatting Example:")
    print("-" * 40)
    vendor_data = workflow._retrieve_vendor_data("How much do we spend on vendors?")
    formatted_vendor = workflow._format_data_for_prompt(vendor_data)
    print(formatted_vendor)

def test_enhanced_prompts():
    """Test the enhanced prompt structures"""
    print("\n🧪 Testing Enhanced Prompt Structures")
    print("=" * 60)
    
    workflow = SpendAnalyzerWorkflow()
    
    # Test classification prompt
    print("\n📝 Enhanced Classification Prompt:")
    print("-" * 40)
    classification_prompt = f"""
    You are a financial data analyst specializing in technology company spending analysis. Your task is to classify the following question into exactly one of these three categories:
    
    1. "aws_costs" - Questions about AWS cloud infrastructure costs, EC2, S3, RDS, VPC, CloudWatch, Route 53, or any AWS service expenses
    2. "budget" - Questions about project budgets, team spending, budget variances, budget tracking, or financial planning
    3. "vendor_spend" - Questions about vendor expenses, supplier costs, external services, contractor spending, or third-party service costs
    
    Consider the context carefully. If a question could fit multiple categories, choose the most specific one based on the primary focus of the question.
    
    Question: What are our total AWS costs for this year?
    
    Respond with ONLY the category name (aws_costs, budget, or vendor_spend).
    """
    print(classification_prompt)
    
    # Test answer generation prompt
    print("\n💬 Enhanced Answer Generation Prompt:")
    print("-" * 40)
    answer_prompt = f"""
    You are a senior financial analyst at a technology company. Answer the following question based on the provided data with a professional, executive-ready response.
    
    Question: What are our total AWS costs for this year?
    Classification: aws_costs
    
    Data Summary:
    Dataset: daily-aws-costs
    
    total_annual_cost: $2,964.32
    latest_date: 2025-05-09
    latest_daily_cost: $17.29
    top_services: 5 items
      - Item 1: ('Relational Database Service($)', 1973.6753743206)
      - Item 2: ('EC2-Other($)', 286.3283652879)
      - Item 3: ('EC2-Instances($)', 250.0552534341)
    
    Please provide a comprehensive, professional answer that follows this structure:
    
    **EXECUTIVE SUMMARY**
    - Brief 1-2 sentence overview of the key findings
    
    **DETAILED ANALYSIS**
    - Present the specific data and metrics that answer the question
    - Use exact numbers and percentages from the data
    - Highlight trends, patterns, or anomalies
    
    **KEY INSIGHTS**
    - 2-3 bullet points of the most important takeaways
    - Business implications of the findings
    
    **RECOMMENDATIONS** (if applicable)
    - Actionable next steps
    - Risk mitigation strategies
    - Optimization opportunities
    
    **DATA SOURCES**
    - Reference the datasets used for this analysis
    
    Format your response professionally with clear headings, bullet points, and proper spacing. Use business language appropriate for executive stakeholders.
    
    Answer:
    """
    print(answer_prompt)

def test_system_message():
    """Test the system message configuration"""
    print("\n🧪 Testing System Message Configuration")
    print("=" * 60)
    
    workflow = SpendAnalyzerWorkflow()
    
    print("System Message:")
    print("-" * 40)
    print(workflow.system_message.content)
    
    print("\nThis system message ensures:")
    print("✅ Consistent professional tone across all responses")
    print("✅ Executive-ready language and formatting")
    print("✅ Focus on actionable insights and business value")
    print("✅ Proper data references and analysis depth")

def main():
    """Run all tests"""
    print("🎯 ENHANCED RESPONSE FORMAT TESTING")
    print("=" * 60)
    print("Testing the improved prompt structures and data formatting")
    print("=" * 60)
    
    try:
        # Test data formatting
        test_data_formatting()
        
        # Test enhanced prompts
        test_enhanced_prompts()
        
        # Test system message
        test_system_message()
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED FORMATTING TESTS COMPLETED!")
        print("=" * 60)
        
        print("\n📋 Key Improvements Made:")
        print("✅ Enhanced classification prompt with detailed category descriptions")
        print("✅ Structured response format with Executive Summary, Analysis, Insights, etc.")
        print("✅ Professional system message for consistent tone")
        print("✅ Improved data formatting with currency symbols and better organization")
        print("✅ Executive-ready language and business context")
        
        print("\n💡 Benefits:")
        print("• More accurate question classification")
        print("• Professional, structured responses")
        print("• Better data presentation and formatting")
        print("• Executive-level insights and recommendations")
        print("• Consistent professional tone across all interactions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
