#!/usr/bin/env python3
"""
Test script to verify individual components work correctly
"""

import os
import sys
from pathlib import Path

def test_dataloader():
    """Test the DataLoader component"""
    print("🧪 Testing DataLoader...")
    
    try:
        from dataloader import DataLoader
        
        # Test initialization
        loader = DataLoader()
        print("✅ DataLoader initialized successfully")
        
        # Test data loading
        data = loader.load_all_data()
        print(f"✅ Loaded {len(data)} datasets")
        
        # Test available datasets
        datasets = loader.get_available_datasets()
        print(f"✅ Available datasets: {datasets}")
        
        # Test getting specific dataframe
        for dataset_name in datasets:
            df = loader.get_dataframe(dataset_name)
            print(f"✅ Retrieved {dataset_name}: {df.shape}")
            
        # Test dataframe info
        for dataset_name in datasets[:1]:  # Test with first dataset
            info = loader.get_dataframe_info(dataset_name)
            print(f"✅ Dataset info for {dataset_name}: {info['shape']}")
            
        print("✅ DataLoader tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {str(e)}")
        return False

def test_workflow_import():
    """Test that the workflow can be imported"""
    print("🧪 Testing Workflow Import...")
    
    try:
        from langgraph_workflow import SpendAnalyzerWorkflow
        print("✅ Workflow import successful")
        return True
        
    except Exception as e:
        print(f"❌ Workflow import failed: {str(e)}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    required_packages = [
        'langgraph',
        'langchain',
        'langchain_openai',
        'pandas',
        'openai',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True

def test_data_files():
    """Test that required data files exist"""
    print("🧪 Testing Data Files...")
    
    docs_path = Path("docs")
    if not docs_path.exists():
        print("❌ docs/ folder not found")
        return False
    
    required_files = [
        "daily-aws-costs.csv",
        "sample-budget-tracking.csv", 
        "sample-vendor-data.csv"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = docs_path / file_name
        if file_path.exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required data files found")
        return True

def main():
    """Run all tests"""
    print("🚀 Starting Component Tests")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_files),
        ("DataLoader", test_dataloader),
        ("Workflow Import", test_workflow_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: python run_workflow.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the workflow.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
