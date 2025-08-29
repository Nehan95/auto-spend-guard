#!/usr/bin/env python3
"""
Enhanced Workflow Visualizer for the updated LangGraph workflow with retriever agent.
This creates HTML visualizations showing the workflow structure, retriever agent integration,
and tool routing capabilities.
"""

import os
from datetime import datetime

class WorkflowVisualizer:
    """Enhanced workflow visualizer with retriever agent and tool routing visualization"""
    
    @staticmethod
    def create_workflow_html(save_path: str = "enhanced_workflow_graph.html") -> str:
        """Create an enhanced HTML visualization of the workflow with retriever agent"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Auto-Spend Workflow with Retriever Agent</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .workflow-section {{
            padding: 30px;
        }}
        .workflow-diagram {{
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 30px 0;
        }}
        .workflow-step {{
            background: white;
            border: 3px solid #3498db;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
            min-width: 300px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .workflow-step:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }}
        .workflow-step.classify {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
        }}
        .workflow-step.routing {{
            border-color: #f39c12;
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
        }}
        .workflow-step.retriever {{
            border-color: #9b59b6;
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
            color: white;
        }}
        .workflow-step.tool {{
            border-color: #27ae60;
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
        }}
        .workflow-step.generate {{
            border-color: #e67e22;
            background: linear-gradient(135deg, #e67e22, #d35400);
            color: white;
        }}
        .workflow-step h3 {{
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }}
        .workflow-step p {{
            margin: 5px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .arrow {{
            font-size: 2em;
            color: #3498db;
            margin: 10px 0;
        }}
        .conditional-routing {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .route-option {{
            background: white;
            border: 2px solid #f39c12;
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
            text-align: center;
            min-width: 200px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .route-option h4 {{
            margin: 0 0 10px 0;
            color: #f39c12;
        }}
        .retriever-agent-section {{
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: center;
        }}
        .retriever-agent-section h2 {{
            margin: 0 0 20px 0;
            font-size: 2em;
        }}
        .agent-features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .feature {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        .feature h4 {{
            margin: 0 0 10px 0;
            color: #f1c40f;
        }}
        .tool-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        .tool-section h2 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }}
        .tools-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .tool-card {{
            background: white;
            border: 2px solid #27ae60;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .tool-card h4 {{
            margin: 0 0 10px 0;
            color: #27ae60;
        }}
        .tool-card p {{
            margin: 5px 0;
            color: #666;
        }}
        .performance-section {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        .performance-section h2 {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric h4 {{
            margin: 0 0 10px 0;
            color: #3498db;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .highlight {{
            background: linear-gradient(135deg, #f1c40f, #f39c12);
            color: #2c3e50;
            padding: 2px 8px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.8em;
        }}
        @media (max-width: 768px) {{
            .conditional-routing {{
                flex-direction: column;
            }}
            .workflow-step {{
                min-width: 250px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Auto-Spend Workflow</h1>
            <p>Intelligent Financial Analysis with Retriever Agent & Tool Routing</p>
            <p><span class="highlight">Updated: {datetime.now().strftime('%B %d, %Y')}</span></p>
        </div>
        
        <div class="workflow-section">
            <h2>🔄 Complete Workflow Flow</h2>
            <div class="workflow-diagram">
                <div class="workflow-step classify">
                    <h3>📝 Question Classification</h3>
                    <p>Analyzes user question and classifies into categories</p>
                    <p><strong>Categories:</strong> aws_costs, budget, vendor_spend</p>
                    <p><strong>Output:</strong> Classification + Message State</p>
                </div>
                
                <div class="arrow">⬇️</div>
                
                <div class="workflow-step routing">
                    <h3>🔄 Conditional Routing</h3>
                    <p>Routes workflow based on classification</p>
                    <p><strong>Routing Logic:</strong> Classification-based decision tree</p>
                </div>
                
                <div class="arrow">⬇️</div>
                
                <div class="workflow-step retriever">
                    <h3>🤖 Retriever Agent</h3>
                    <p>Intelligent tool selection and execution</p>
                    <p><strong>Capabilities:</strong> Context-aware routing, fallback handling</p>
                </div>
                
                <div class="arrow">⬇️</div>
                
                <div class="workflow-step tool">
                    <h3>🔧 Tool Execution</h3>
                    <p>Executes selected tools and retrieves data</p>
                    <p><strong>Methods:</strong> Agent-based or direct execution</p>
                </div>
                
                <div class="arrow">⬇️</div>
                
                <div class="workflow-step generate">
                    <h3>💡 Answer Generation</h3>
                    <p>Generates comprehensive financial analysis</p>
                    <p><strong>Features:</strong> Anomaly detection, insights, recommendations</p>
                </div>
            </div>
            
            <h3>🔄 Conditional Routing Details</h3>
            <div class="conditional-routing">
                <div class="route-option">
                    <h4>🔴 AWS Costs</h4>
                    <p>Route: classify_question → retrieve_aws_data</p>
                    <p>Tool: AWS Data Retrieval</p>
                    <p>Data: Cloud infrastructure costs</p>
                </div>
                <div class="route-option">
                    <h4>🟡 Budget Tracking</h4>
                    <p>Route: classify_question → retrieve_budget_data</p>
                    <p>Tool: Budget Data Retrieval</p>
                    <p>Data: Project budgets & variances</p>
                </div>
                <div class="route-option">
                    <h4>🟢 Vendor Spending</h4>
                    <p>Route: classify_question → retrieve_vendor_data</p>
                    <p>Tool: Vendor Data Retrieval</p>
                    <p>Data: Vendor costs & contracts</p>
                </div>
            </div>
        </div>
        
        <div class="retriever-agent-section">
            <h2>🤖 Retriever Agent Architecture</h2>
            <p>The retriever agent intelligently routes classification responses to specific tools</p>
            
            <div class="agent-features">
                <div class="feature">
                    <h4>🧠 Intelligent Routing</h4>
                    <p>Uses classification and context to determine optimal tool selection</p>
                </div>
                <div class="feature">
                    <h4>🔄 Fallback Strategy</h4>
                    <p>Automatically falls back to direct tool execution if agent fails</p>
                </div>
                <div class="feature">
                    <h4>📊 Tool Usage Tracking</h4>
                    <p>Records which tools were executed and how many times</p>
                </div>
                <div class="feature">
                    <h4>⚡ Performance Monitoring</h4>
                    <p>Tracks agent execution time and efficiency metrics</p>
                </div>
            </div>
        </div>
        
        <div class="tool-section">
            <h2>🔧 Data Retrieval Tools</h2>
            <div class="tools-grid">
                <div class="tool-card">
                    <h4>☁️ AWS Data Retrieval</h4>
                    <p><strong>Purpose:</strong> Cloud infrastructure cost analysis</p>
                    <p><strong>Data:</strong> Daily costs, service breakdowns, trends</p>
                    <p><strong>Features:</strong> Anomaly detection, cost optimization</p>
                </div>
                <div class="tool-card">
                    <h4>💰 Budget Data Retrieval</h4>
                    <p><strong>Purpose:</strong> Project budget tracking and analysis</p>
                    <p><strong>Data:</strong> Budget variances, team spending, status</p>
                    <p><strong>Features:</strong> Variance analysis, project monitoring</p>
                </div>
                <div class="tool-card">
                    <h4>🏢 Vendor Data Retrieval</h4>
                    <p><strong>Purpose:</strong> Vendor spending and contract analysis</p>
                    <p><strong>Data:</strong> Contract details, risk assessment, spending patterns</p>
                    <p><strong>Features:</strong> Risk analysis, contract monitoring</p>
                </div>
            </div>
        </div>
        
        <div class="performance-section">
            <h2>⚡ Enhanced Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <h4>📊 Classification</h4>
                    <p>Question analysis time</p>
                    <p>Token usage tracking</p>
                    <p>Classification accuracy</p>
                </div>
                <div class="metric">
                    <h4>🤖 Retriever Agent</h4>
                    <p>Agent execution time</p>
                    <p>Tool routing decisions</p>
                    <p>Fallback frequency</p>
                </div>
                <div class="metric">
                    <h4>🔧 Tool Execution</h4>
                    <p>Tool performance time</p>
                    <p>Data retrieval efficiency</p>
                    <p>Tool usage analytics</p>
                </div>
                <div class="metric">
                    <h4>💡 Answer Generation</h4>
                    <p>LLM processing time</p>
                    <p>Response quality metrics</p>
                    <p>Token cost analysis</p>
                </div>
            </div>
        </div>
        
        <div class="workflow-section">
            <h2>📈 State Management & Tracking</h2>
            <div class="tools-grid">
                <div class="tool-card">
                    <h4>💬 Message State</h4>
                    <p><strong>Purpose:</strong> Complete workflow progression tracking</p>
                    <p><strong>Content:</strong> Classification, tool execution, answer generation</p>
                    <p><strong>Benefits:</strong> Audit trail, debugging, performance analysis</p>
                </div>
                <div class="tool-card">
                    <h4>🔧 Tool Usage Tracking</h4>
                    <p><strong>Purpose:</strong> Record which tools were executed</p>
                    <p><strong>Content:</strong> Tool names, execution counts, performance data</p>
                    <p><strong>Benefits:</strong> Optimization insights, cost analysis</p>
                </div>
                <div class="tool-card">
                    <h4>⚡ Latency Metrics</h4>
                    <p><strong>Purpose:</strong> Comprehensive performance monitoring</p>
                    <p><strong>Content:</strong> Step-by-step timing, efficiency analysis</p>
                    <p><strong>Benefits:</strong> Performance optimization, bottleneck identification</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 Enhanced Auto-Spend Workflow with Retriever Agent</p>
            <p>Intelligent Financial Analysis • Tool Routing • Performance Monitoring</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Enhanced workflow visualization saved to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating workflow visualization: {str(e)}")
            return None
    
    @staticmethod
    def display_text_workflow():
        """Display a text-based representation of the enhanced workflow"""
        print("\n" + "="*80)
        print("🚀 ENHANCED AUTO-SPEND WORKFLOW WITH RETRIEVER AGENT")
        print("="*80)
        
        workflow_structure = """
        START
          ↓
    ┌─────────────────┐
    │ Classify        │ ← question + classification
    │ Question        │
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Conditional     │ ← routing based on classification
    │ Routing         │
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Retriever       │ ← intelligent tool selection
    │ Agent           │
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Tool Execution  │ ← agent-based or direct execution
    │ & Data Retrieval│
    └─────────────────┘
          ↓
    ┌─────────────────┐
    │ Generate        │ ← comprehensive analysis
    │ Answer          │
    └─────────────────┘
          ↓
         END
        """
        
        print(workflow_structure)
        
        print("\n🔄 CONDITIONAL ROUTING PATHS:")
        print("  • aws_costs → retrieve_aws_data → AWS Data Retrieval Tool")
        print("  • budget → retrieve_budget_data → Budget Data Retrieval Tool")
        print("  • vendor_spend → retrieve_vendor_data → Vendor Data Retrieval Tool")
        
        print("\n🤖 RETRIEVER AGENT FEATURES:")
        print("  • Intelligent tool routing based on classification")
        print("  • Context-aware tool selection")
        print("  • Automatic fallback to direct tool execution")
        print("  • Tool usage tracking and analytics")
        print("  • Performance monitoring and optimization")
        
        print("\n🔧 DATA RETRIEVAL TOOLS:")
        print("  • AWS Data Retrieval: Cloud infrastructure cost analysis")
        print("  • Budget Data Retrieval: Project budget tracking and variance analysis")
        print("  • Vendor Data Retrieval: Vendor spending and contract analysis")
        
        print("\n📊 ENHANCED PERFORMANCE METRICS:")
        print("  • Classification performance and token usage")
        print("  • Retriever agent execution time and efficiency")
        print("  • Tool execution performance and data retrieval rates")
        print("  • Answer generation quality and cost analysis")
        
        print("\n💬 STATE MANAGEMENT:")
        print("  • Message state tracking throughout workflow")
        print("  • Tool usage recording and analytics")
        print("  • Complete audit trail and debugging information")
        
        print("="*80)

if __name__ == "__main__":
    # Create enhanced workflow visualization
    visualizer = WorkflowVisualizer()
    
    # Generate HTML visualization
    html_file = visualizer.create_workflow_html()
    
    if html_file:
        print(f"🎉 Enhanced workflow visualization created successfully!")
        print(f"📁 File: {html_file}")
        print(f"🌐 Open in your browser to view the interactive workflow diagram")
    else:
        print("📊 Displaying text-based workflow structure instead:")
        visualizer.display_text_workflow()
