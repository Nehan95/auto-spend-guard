#!/usr/bin/env python3
"""
Workflow Visualizer Module
Handles the creation of HTML-based workflow visualizations
"""

import os
import webbrowser

class WorkflowVisualizer:
    """Creates HTML-based visualizations of LangGraph workflows"""
    
    @staticmethod
    def create_workflow_html(save_path: str = "workflow_graph.html") -> str:
        """Create a visual representation of the workflow graph using HTML"""
        try:
            # Create HTML-based workflow visualization
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Auto-Spend LangGraph Workflow</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .workflow-title { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .workflow-diagram { display: flex; justify-content: space-between; align-items: center; margin: 40px 0; }
        .node { 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center; 
            color: white; 
            font-weight: bold; 
            min-width: 120px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .start-end { background: linear-gradient(135deg, #27ae60, #2ecc71); }
        .classify { background: linear-gradient(135deg, #3498db, #5dade2); }
        .retrieve { background: linear-gradient(135deg, #f39c12, #f7dc6f); }
        .generate { background: linear-gradient(135deg, #e74c3c, #ec7063); }
        .arrow { font-size: 24px; color: #7f8c8d; margin: 0 20px; }
        .data-flow { text-align: center; margin: 20px 0; color: #7f8c8d; font-style: italic; }
        .details { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .details h3 { color: #2c3e50; margin-top: 0; }
        .details ul { margin: 10px 0; }
        .details li { margin: 5px 0; }
        .node-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .node-card { background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .node-card h4 { margin-top: 0; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <div class="workflow-title">
            <h1>🔧 Auto-Spend LangGraph Workflow</h1>
            <p>Financial Data Analysis and Anomaly Detection System</p>
        </div>
        
        <div class="workflow-diagram">
            <div class="node start-end">START</div>
            <div class="arrow">→</div>
            <div class="node classify">Classify<br>Question</div>
            <div class="arrow">→</div>
            <div class="node retrieve">🤖 Agent<br>Data Retrieval</div>
            <div class="arrow">→</div>
            <div class="node generate">Generate<br>Answer</div>
            <div class="arrow">→</div>
            <div class="node start-end">END</div>
        </div>
        
        <div class="data-flow">
            <strong>Data Flow:</strong> question → classification → agent_analysis → final_answer
        </div>
        
        <div class="details">
            <h3>🔄 Workflow Process</h3>
            <ul>
                <li><strong>START → Classify Question:</strong> User inputs a financial question</li>
                <li><strong>Classify → Agent Data Retrieval:</strong> AI determines category and intelligent agent selects relevant data</li>
                <li><strong>Agent → Generate Answer:</strong> Agent analyzes data and provides comprehensive insights</li>
                <li><strong>Generate → END:</strong> Professional, structured response with anomaly detection</li>
            </ul>
        </div>
        
        <div class="node-details">
            <div class="node-card">
                <h4>📝 Classify Question</h4>
                <p>Uses OpenAI to categorize questions into:</p>
                <ul>
                    <li><strong>aws_costs:</strong> Cloud infrastructure expenses</li>
                    <li><strong>budget:</strong> Project and team spending</li>
                    <li><strong>vendor_spend:</strong> External service costs</li>
                </ul>
            </div>
            
            <div class="node-card">
                <h4>🤖 Agent Data Retrieval</h4>
                <p>Intelligent agent with specialized tools:</p>
                <ul>
                    <li>AWS cost analysis tool</li>
                    <li>Budget tracking tool</li>
                    <li>Vendor spending tool</li>
                    <li>Anomaly detection</li>
                    <li>Trend analysis</li>
                </ul>
            </div>
            
            <div class="node-card">
                <h4>💬 Generate Answer</h4>
                <p>Creates professional responses with:</p>
                <ul>
                    <li>Executive summary</li>
                    <li>Detailed analysis</li>
                    <li>Anomaly detection</li>
                    <li>Actionable insights</li>
                </ul>
            </div>
        </div>
        
        <div class="details">
            <h3>📊 Supported Data Sources</h3>
            <ul>
                <li><strong>AWS Costs:</strong> Daily cloud service expenses, service breakdowns, cost trends</li>
                <li><strong>Budget Tracking:</strong> Project budgets, team spending, variance analysis</li>
                <li><strong>Vendor Spending:</strong> Contract details, risk assessment, spending patterns</li>
            </ul>
        </div>
    </div>
</body>
</html>
            """
            
            # Save HTML file
            with open(save_path, 'w') as f:
                f.write(html_content)
            
            print(f"\n🎨 Workflow visualization saved as: {save_path}")
            print("💡 Open this HTML file in your browser to see the interactive workflow diagram!")
            print("🌐 You can also share this file with team members for documentation")
            
            # Try to open the file in the default browser
            try:
                file_path = os.path.abspath(save_path)
                webbrowser.open(f'file://{file_path}')
                print("🚀 Opened workflow visualization in your default browser!")
            except Exception as e:
                print(f"💡 To view the visualization, open {save_path} in your web browser")
            
            return save_path
            
        except Exception as e:
            print(f"\n❌ Error creating HTML visualization: {str(e)}")
            return None
    
    @staticmethod
    def display_text_workflow():
        """Display a text-based representation of the workflow"""
        print("\n" + "="*60)
        print("📊 TEXT-BASED WORKFLOW STRUCTURE")
        print("="*60)
        
        workflow_structure = """
        START
          ↓
    ┌─────────────────┐
    │ Classify        │ ← question
    │ Question        │
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Retrieve        │ ← classification
    │ Data            │
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Generate        │ ← relevant_data
    │ Answer          │
    └─────────────────┘
          ↓
         END
        """
        
        print(workflow_structure)
        
        # Add detailed flow information
        print("\n🔄 WORKFLOW FLOW DETAILS:")
        print("   • START → Classify Question: User input question")
        print("   • Classify → Retrieve Data: Based on classification (aws_costs/budget/vendor_spend)")
        print("   • Retrieve → Generate Answer: Using retrieved data and LLM analysis")
        print("   • Generate → END: Final structured response")
        
        print("\n📊 DATA FLOW:")
        print("   • question: User's financial question")
        print("   • classification: AI-determined category")
        print("   • relevant_data: Retrieved financial data")
        print("   • final_answer: Professional, structured response")
        
        print("="*60)
