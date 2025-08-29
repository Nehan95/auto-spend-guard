#!/usr/bin/env python3
"""
Debug Workflow Object Script
Explores the compiled workflow object to understand its structure
"""

from langgraph_workflow import SpendAnalyzerWorkflow

def main():
    """Debug the workflow object"""
    print("🔍 DEBUGGING WORKFLOW OBJECT STRUCTURE")
    print("=" * 60)
    
    try:
        # Initialize the workflow
        workflow = SpendAnalyzerWorkflow()
        
        # Get the compiled workflow
        compiled = workflow.workflow
        
        print(f"📊 Compiled Workflow Type: {type(compiled)}")
        print(f"📊 Compiled Workflow Class: {compiled.__class__.__name__}")
        
        # List all attributes
        print(f"\n🔍 ALL ATTRIBUTES:")
        for attr in dir(compiled):
            if not attr.startswith('_'):
                try:
                    value = getattr(compiled, attr)
                    if callable(value):
                        print(f"   • {attr}(): {type(value).__name__}")
                    else:
                        print(f"   • {attr}: {type(value).__name__} = {value}")
                except Exception as e:
                    print(f"   • {attr}: Error accessing - {e}")
        
        # Try to access specific attributes
        print(f"\n🔍 SPECIFIC ATTRIBUTES:")
        
        if hasattr(compiled, 'nodes'):
            print(f"   • nodes: {compiled.nodes}")
            if compiled.nodes:
                print(f"     - Keys: {list(compiled.nodes.keys())}")
                print(f"     - Types: {[type(node).__name__ for node in compiled.nodes.values()]}")
        
        if hasattr(compiled, 'state_schema'):
            print(f"   • state_schema: {compiled.state_schema}")
        
        if hasattr(compiled, 'config'):
            print(f"   • config: {compiled.config}")
        
        if hasattr(compiled, 'graph'):
            print(f"   • graph: {compiled.graph}")
        
        print("\n" + "="*60)
        print("✅ Debug completed!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error debugging workflow: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
