from typing import Dict, Any, List, TypedDict, Union
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from dataloader import DataLoader
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import tiktoken

# Load environment variables
load_dotenv()

# Define the state structure
class WorkflowState(TypedDict):
    question: str
    classification: str
    relevant_data: Dict[str, Any]
    final_answer: str
    error: str
    latency_metrics: Dict[str, Any]
    messages: List[BaseMessage]  # Message state for tracking tools
    tools_used: List[str]  # Track which tools were executed

# Data Retrieval Tools
class DataRetrievalTool(BaseTool):
    """Base tool for data retrieval operations"""
    name: str
    description: str
    data_loader: DataLoader = None
    
    def _run(self, query: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement _run")
    
    async def _arun(self, query: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement _arun")

class RetrieverAgent:
    """Intelligent agent for routing classification responses to specific tools and retrieving relevant data"""
    
    def __init__(self, llm: ChatOpenAI, tools: List[DataRetrievalTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = {tool.name: tool.description for tool in tools}
        
        # Create the retriever prompt template
        self.retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent data retrieval agent that routes user queries to the most appropriate tools based on classification and context.

Available tools:
{tool_descriptions}

Your task is to:
1. Analyze the user's question and its classification
2. Determine which tool(s) are most relevant
3. Execute the appropriate tool(s) to retrieve data
4. Return comprehensive, relevant data with analysis

Guidelines:
- Use the classification to guide tool selection
- Consider the specific context and requirements of the question
- Execute tools efficiently and return structured data
- Provide insights and analysis based on the retrieved data
- Handle cases where multiple tools might be relevant

Always return data in a structured format that includes:
- Data source and tool used
- Key metrics and insights
- Anomaly detection
- Business implications
- Recommendations"""),
            ("human", "Question: {question}\nClassification: {classification}\n\nPlease retrieve relevant data using the most appropriate tools."),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=list(tools),
            prompt=self.retriever_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=list(tools),
            verbose=True,
            handle_parsing_errors=True
        )
    
    def retrieve_data(self, question: str, classification: str) -> Dict[str, Any]:
        """Retrieve relevant data using intelligent tool routing"""
        try:
            # Format tool descriptions for the prompt
            tool_descriptions = []
            for name, description in self.tool_descriptions.items():
                tool_descriptions.append(f"- {name}: {description}")
            
            tool_descriptions_text = "\n".join(tool_descriptions)
            
            # Execute the agent
            response = self.agent_executor.invoke({
                "question": question,
                "classification": classification,
                "tool_descriptions": tool_descriptions_text
            })
            
            # Extract the response
            if "output" in response:
                return {
                    "success": True,
                    "data": response["output"],
                    "tool_execution": response.get("intermediate_steps", []),
                    "agent_response": response
                }
            else:
                return {
                    "success": False,
                    "error": "Unexpected agent response format",
                    "raw_response": response
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error in retriever agent: {str(e)}",
                "exception": str(e)
            }
    
    def get_tool_usage_summary(self, tool_execution: List) -> Dict[str, Any]:
        """Extract tool usage summary from agent execution"""
        if not tool_execution:
            return {}
        
        tool_summary = {}
        for step in tool_execution:
            if len(step) >= 2:
                tool_name = step[0].tool if hasattr(step[0], 'tool') else str(step[0])
                tool_input = step[0].tool_input if hasattr(step[0], 'tool_input') else str(step[0])
                tool_output = step[1] if len(step) > 1 else "No output"
                
                if tool_name not in tool_summary:
                    tool_summary[tool_name] = {
                        "executions": 0,
                        "inputs": [],
                        "outputs": []
                    }
                
                tool_summary[tool_name]["executions"] += 1
                tool_summary[tool_name]["inputs"].append(tool_input)
                tool_summary[tool_name]["outputs"].append(tool_output)
        
        return tool_summary

class AWSDataRetrievalTool(DataRetrievalTool):
    """Tool for retrieving AWS cost data"""
    name: str = "aws_data_retrieval"
    description: str = "Retrieves AWS cost data including daily costs, service breakdowns, and cost trends"
    
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader=data_loader)
    
    def _run(self, query: str, **kwargs) -> str:
        try:
            df = self.data_loader.get_dataframe("daily-aws-costs")
            if df is None:
                return "No AWS cost data available"
            
            # Get the total costs row (Service total)
            total_row = df[df['Service'] == 'Service total']
            if total_row.empty:
                return "No total costs data available"
            
            # Get total annual cost
            total_annual_cost = float(total_row['Total costs($)'].iloc[0])
            
            # Get daily data (exclude Service total row)
            daily_data = df[df['Service'] != 'Service total'].copy()
            
            # Convert Service column to datetime for daily analysis
            daily_data['Date'] = pd.to_datetime(daily_data['Service'], errors='coerce')
            daily_data = daily_data.dropna(subset=['Date'])
            
            if daily_data.empty:
                return "No daily cost data available"
            
            # Calculate daily cost statistics
            daily_costs = daily_data['Total costs($)'].astype(float)
            avg_daily_cost = daily_costs.mean()
            max_daily_cost = daily_costs.max()
            min_daily_cost = daily_costs.min()
            
            # Get recent trends (last 7 days)
            recent_data = daily_data.tail(7)
            recent_avg = recent_data['Total costs($)'].astype(float).mean()
            
            # Get service breakdown from total row
            service_columns = [col for col in df.columns if col not in ['Service', 'Total costs($)']]
            service_breakdown = []
            for service in service_columns:
                cost = total_row[service].iloc[0]
                if cost and str(cost).strip() != '':
                    try:
                        service_cost = float(cost)
                        service_breakdown.append((service.replace('($)', ''), service_cost))
                    except:
                        continue
            
            # Sort services by cost
            service_breakdown.sort(key=lambda x: x[1], reverse=True)
            top_services = service_breakdown[:5]
            
            result = f"""
AWS Cost Data Analysis:
- Total Annual Cost: ${total_annual_cost:,.2f}
- Average Daily Cost: ${avg_daily_cost:.2f}
- Daily Cost Range: ${min_daily_cost:.2f} - ${max_daily_cost:.2f}
- Recent 7-day Average: ${recent_avg:.2f}

Top Services by Cost:
{chr(10).join([f"  • {service}: ${cost:.2f}" for service, cost in top_services])}

Data Shape: {df.shape[0]} rows, {df.shape[1]} columns
            """
            return result.strip()
        except Exception as e:
            return f"Error retrieving AWS data: {str(e)}"

class BudgetDataRetrievalTool(DataRetrievalTool):
    """Tool for retrieving budget tracking data"""
    name: str = "budget_data_retrieval"
    description: str = "Retrieves budget tracking data including project budgets, team spending, and variance analysis"
    
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader=data_loader)
    
    def _run(self, query: str, **kwargs) -> str:
        try:
            df = self.data_loader.get_dataframe("sample-budget-tracking")
            if df is None:
                return "No budget tracking data available"
            
            # Basic statistics - using correct column names
            total_budget = df['Monthly Budget'].astype(float).sum()
            total_spent = df['Actual Spend'].astype(float).sum()
            variance = total_budget - total_spent
            variance_pct = (variance / total_budget) * 100 if total_budget > 0 else 0
            
            # Project status breakdown
            status_counts = df['Status'].value_counts()
            
            # Top projects by budget
            top_projects = df.nlargest(5, 'Monthly Budget')[['Project', 'Monthly Budget', 'Actual Spend', 'Status']]
            
            # Team breakdown
            team_budgets = df.groupby('Team')['Monthly Budget'].sum().sort_values(ascending=False)
            
            # Over budget projects
            over_budget = df[df['Variance %'].str.contains('-', na=False)]
            over_budget_count = len(over_budget)
            
            result = f"""
Budget Tracking Analysis:
- Total Monthly Budget: ${total_budget:,.2f}
- Total Actual Spend: ${total_spent:,.2f}
- Variance: ${variance:,.2f} ({variance_pct:+.1f}%)
- Projects Over Budget: {over_budget_count}

Project Status Breakdown:
{status_counts.to_string()}

Top 5 Projects by Budget:
{top_projects.to_string()}

Team Budget Allocation:
{team_budgets.to_string()}

Data Shape: {df.shape[0]} rows, {df.shape[1]} columns
            """
            return result.strip()
        except Exception as e:
            return f"Error retrieving budget data: {str(e)}"

class VendorDataRetrievalTool(DataRetrievalTool):
    """Tool for retrieving vendor spending data"""
    name: str = "vendor_data_retrieval"
    description: str = "Retrieves vendor spending data including contract details, risk assessment, and spending patterns"
    
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader=data_loader)
    
    def _run(self, query: str, **kwargs) -> str:
        try:
            # Try CSV first, then JSON
            df = self.data_loader.get_dataframe("sample-vendor-data")
            if df is None:
                df = self.data_loader.get_dataframe("sample-vendor-data-json")
            
            if df is None:
                return "No vendor data available"
            
            # Basic statistics - using correct column names
            total_budget = df['Annual Budget Approved'].astype(float).sum()
            total_spend = df['Current Annual Spend'].astype(float).sum()
            budget_utilization = (total_spend/total_budget*100) if total_budget > 0 else 0
            
            # Risk level breakdown
            risk_counts = df['Risk Level'].value_counts()
            risk_analysis = f"\nRisk Level Breakdown:\n{risk_counts.to_string()}"
            
            # Vendor types
            type_counts = df['Vendor Type'].value_counts()
            type_analysis = f"\nVendor Type Breakdown:\n{type_counts.to_string()}"
            
            # Status breakdown
            status_counts = df['Status'].value_counts()
            status_analysis = f"\nStatus Breakdown:\n{status_counts.to_string()}"
            
            # Top vendors by budget
            top_vendors = df.nlargest(5, 'Annual Budget Approved')[['Vendor Name', 'Vendor Type', 'Annual Budget Approved', 'Current Annual Spend', 'Risk Level']]
            
            # Contract analysis
            active_contracts = len(df[df['Status'] == 'Active'])
            total_contracts = len(df)
            
            result = f"""
Vendor Spending Analysis:
- Total Annual Budget: ${total_budget:,.2f}
- Total Current Spend: ${total_spend:,.2f}
- Budget Utilization: {budget_utilization:.1f}%
- Active Contracts: {active_contracts}/{total_contracts}

{risk_analysis}
{type_analysis}
{status_analysis}

Top 5 Vendors by Budget:
{top_vendors.to_string()}

Data Shape: {df.shape[0]} rows, {df.shape[1]} columns
            """
            return result.strip()
        except Exception as e:
            return f"Error retrieving vendor data: {str(e)}"

class SpendAnalyzerWorkflow:
    """LangGraph workflow for analyzing spending data with classification and retrieval"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Set system message for consistent behavior
        self.system_message = SystemMessage(content="""You are a senior financial analyst at a company. 
        You specialize in analyzing spending data, budget tracking, and vendor management. 
        You have expertise in anomaly detection, statistical analysis, and identifying unusual patterns in financial data.
        Always provide professional, executive-ready responses with clear structure, specific data references, 
        actionable insights, and comprehensive anomaly detection. Use business language appropriate for stakeholders and decision-makers.
        When analyzing data, pay special attention to outliers, unusual spikes or drops, and patterns that deviate from normal trends.""")
        self.data_loader = DataLoader()
        
        # Load all data
        self.data_loader.load_all_data()
        
        # Initialize data retrieval tools
        self.aws_tool = AWSDataRetrievalTool(self.data_loader)
        self.budget_tool = BudgetDataRetrievalTool(self.data_loader)
        self.vendor_tool = VendorDataRetrievalTool(self.data_loader)
        
        # Create the retriever agent
        self.retriever_agent = RetrieverAgent(
            self.llm, 
            [self.aws_tool, self.budget_tool, self.vendor_tool]
        )
        
        # Create the data retrieval agent (kept for backward compatibility)
        self.data_retrieval_agent = self._create_data_retrieval_agent()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_data_retrieval_agent(self) -> AgentExecutor:
        """Create the data retrieval agent with specialized tools"""
        tools = [self.aws_tool, self.budget_tool, self.vendor_tool]
        
        # Create the agent prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized data retrieval agent for financial analysis. Your job is to intelligently retrieve and analyze financial data based on user queries.

Available tools:
- aws_data_retrieval: For AWS cost data analysis
- budget_data_retrieval: For budget tracking and project spending
- vendor_data_retrieval: For vendor spending and contract analysis

When given a query:
1. Determine which data source is most relevant
2. Use the appropriate tool to retrieve data
3. Provide a comprehensive analysis with key insights
4. Identify any anomalies or unusual patterns
5. Format the response clearly for further analysis

Always be thorough and provide actionable insights from the data."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def display_workflow_info(self):
        """Display detailed information about the compiled workflow"""
        print("\n" + "="*60)
        print("🔧 DETAILED WORKFLOW INFORMATION")
        print("="*60)
        
        # Basic workflow info
        print(f"📊 Workflow Class: {self.workflow.__class__.__name__}")
        print(f"🔗 Total Nodes: {len(self.workflow.nodes)}")
        print(f"📝 Workflow Name: {self.workflow.name}")
        
        # Node details
        print(f"\n📝 NODES:")
        for node_name, node in self.workflow.nodes.items():
            print(f"   • {node_name}: {node.__class__.__name__}")
        
        # Channel information
        print(f"\n🔗 CHANNELS:")
        print(f"   • Input Channel: {self.workflow.input_channels}")
        print(f"   • Output Channels: {self.workflow.output_channels}")
        print(f"   • Stream Channels: {self.workflow.stream_channels}")
        
        # Workflow configuration
        print(f"\n⚙️ WORKFLOW CONFIGURATION:")
        print(f"   • Entry Point: classify_question")
        print(f"   • Flow: classify_question → conditional routing → specific data retrieval → generate_answer → END")
        print(f"   • Conditional Routing: Based on question classification (aws_costs, budget, vendor_spend)")
        print(f"   • State Management: WorkflowState with question, classification, relevant_data, final_answer, messages, tools_used")
        print(f"   • Data Retrieval: Direct tool execution based on classification")
        print(f"   • Debug Mode: {self.workflow.debug}")
        
        # Tool information
        print(f"\n🔧 DATA RETRIEVAL TOOLS:")
        print(f"   • AWS Tool: {self.aws_tool.name} - {self.aws_tool.description}")
        print(f"   • Budget Tool: {self.budget_tool.name} - {self.budget_tool.description}")
        print(f"   • Vendor Tool: {self.vendor_tool.name} - {self.vendor_tool.description}")
        print(f"   • Tool Selection: Automatic based on question classification")
        
        # Retriever agent information
        print(f"\n🤖 RETRIEVER AGENT:")
        print(f"   • Agent Type: OpenAI Functions Agent with Intelligent Routing")
        print(f"   • Capabilities: Classification-based tool routing, intelligent data retrieval")
        print(f"   • Fallback Strategy: Direct tool execution if agent fails")
        print(f"   • Tool Integration: Seamless integration with all data retrieval tools")
        print(f"   • Context Awareness: Uses question context and classification for optimal tool selection")
        
        # Message state and tool tracking
        print(f"\n💬 MESSAGE STATE & TOOL TRACKING:")
        print(f"   • Message State: Enabled for tracking workflow progression")
        print(f"   • Tool Usage Tracking: Records which tools were executed")
        print(f"   • Workflow History: Complete message trail from start to finish")
        print(f"   • Performance Metrics: Tool execution times and response lengths")
        print(f"   • Agent Execution Tracking: Retriever agent performance and tool routing decisions")
        
        # Performance monitoring
        print(f"\n⚡ PERFORMANCE MONITORING:")
        print(f"   • Latency Tracking: Enabled for all workflow steps")
        print(f"   • Token Counting: Input/output token tracking for all LLM calls")
        print(f"   • Metrics: Question classification, tool execution, response generation")
        print(f"   • Granularity: Tool vs. overhead timing breakdown")
        print(f"   • Cost Estimation: GPT-4o-mini pricing calculations")
        
        print("="*60)
    
    def visualize_workflow(self, save_path: str = "enhanced_workflow_graph.html"):
        """Create a visual representation of the enhanced workflow graph using HTML"""
        try:
            from workflow_visualizer import WorkflowVisualizer
            
            # Use the enhanced visualizer module
            result = WorkflowVisualizer.create_workflow_html(save_path)
            
            if result is None:
                print("📊 Displaying text-based workflow structure instead:")
                self._display_text_workflow()
            
            return result
            
        except ImportError:
            print("\n⚠️  WorkflowVisualizer module not found. Using text-based display instead:")
            self._display_text_workflow()
        except Exception as e:
            print(f"\n❌ Error creating visualization: {str(e)}")
            print("📊 Displaying text-based workflow structure instead:")
            self._display_text_workflow()
    
    def _display_text_workflow(self):
        """Display a text-based representation of the enhanced workflow"""
        try:
            from workflow_visualizer import WorkflowVisualizer
            WorkflowVisualizer.display_text_workflow()
        except ImportError:
            # Fallback to inline text display if module not available
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
            
            print("="*80)
    
    def display_latency_metrics(self, state: WorkflowState = None):
        """Display latency metrics for workflow performance analysis"""
        if state is None or "latency_metrics" not in state:
            print("\n📊 No latency metrics available. Run the workflow first to see performance data.")
            return
        
        print("\n" + "="*60)
        print("⚡ LATENCY METRICS & PERFORMANCE ANALYSIS")
        print("="*60)
        
        latency_metrics = state["latency_metrics"]
        
        # Question Classification Metrics
        if "question_classification" in latency_metrics:
            qc_metrics = latency_metrics["question_classification"]
            print(f"\n🔍 QUESTION CLASSIFICATION:")
            print(f"   • Total Duration: {qc_metrics.get('total_duration_seconds', 0)}s")
            print(f"   • LLM Duration: {qc_metrics.get('llm_duration_seconds', 0)}s")
            print(f"   • Overhead: {qc_metrics.get('overhead_duration_seconds', 0)}s")
            print(f"   • Input Tokens: {qc_metrics.get('input_tokens', 0)}")
            print(f"   • Output Tokens: {qc_metrics.get('output_tokens', 0)}")
            print(f"   • Total Tokens: {qc_metrics.get('total_tokens', 0)}")
            print(f"   • Status: {qc_metrics.get('status', 'unknown')}")
        
        # Data Retrieval Metrics
        if "data_retrieval" in latency_metrics:
            dr_metrics = latency_metrics["data_retrieval"]
            print(f"\n🔍 DATA RETRIEVAL:")
            print(f"   • Total Duration: {dr_metrics.get('total_duration_seconds', 0)}s")
            print(f"   • Tool Duration: {dr_metrics.get('tool_duration_seconds', 0)}s")
            print(f"   • Overhead: {dr_metrics.get('overhead_duration_seconds', 0)}s")
            print(f"   • Status: {dr_metrics.get('status', 'unknown')}")
            if "retrieval_metadata" in state.get("relevant_data", {}):
                metadata = state["relevant_data"]["retrieval_metadata"]
                print(f"   • Response Length: {metadata.get('response_length', 0)} characters")
                print(f"   • Tools Executed: {metadata.get('tools_executed', 0)}")
                print(f"   • Tool Used: {metadata.get('tool_used', 'unknown')}")
                print(f"   • Data Source: {metadata.get('data_source', 'unknown')}")
        
        # Response Generation Metrics
        if "response_generation" in latency_metrics:
            rg_metrics = latency_metrics["response_generation"]
            print(f"\n🔍 RESPONSE GENERATION:")
            print(f"   • Total Duration: {rg_metrics.get('total_duration_seconds', 0)}s")
            print(f"   • LLM Duration: {rg_metrics.get('llm_duration_seconds', 0)}s")
            print(f"   • Overhead: {rg_metrics.get('overhead_duration_seconds', 0)}s")
            print(f"   • Response Length: {rg_metrics.get('response_length', 0)} characters")
            print(f"   • Input Tokens: {rg_metrics.get('input_tokens', 0)}")
            print(f"   • Output Tokens: {rg_metrics.get('output_tokens', 0)}")
            print(f"   • Total Tokens: {rg_metrics.get('total_tokens', 0)}")
            print(f"   • Status: {rg_metrics.get('status', 'unknown')}")
        
        # Tool Usage and Message State
        if "tools_used" in state:
            print(f"\n🔧 TOOL USAGE:")
            tools_used = state["tools_used"]
            print(f"   • Tools Executed: {', '.join(tools_used) if tools_used else 'None'}")
            print(f"   • Total Tools Used: {len(tools_used)}")
            
            # Tool-specific performance metrics
            if tools_used and "data_retrieval" in latency_metrics:
                dr_metrics = latency_metrics["data_retrieval"]
                
                # Check if retriever agent was used
                relevant_data = state.get("relevant_data", {})
                if "retrieval_method" in relevant_data and "retriever_agent" in relevant_data["retrieval_method"]:
                    print(f"   • Retrieval Method: {relevant_data['data_retrieval_method']}")
                    print(f"   • Agent Execution Time: {dr_metrics.get('agent_duration_seconds', 0):.3f}s")
                    print(f"   • Agent Overhead: {dr_metrics.get('overhead_duration_seconds', 0):.3f}s")
                    
                    # Show tool usage summary if available
                    if "tool_usage_summary" in relevant_data:
                        tool_summary = relevant_data["tool_usage_summary"]
                        print(f"   • Agent Tool Executions: {len(tool_summary)}")
                        for tool_name, tool_info in tool_summary.items():
                            executions = tool_info.get("executions", 0)
                            print(f"     - {tool_name}: {executions} execution(s)")
                    
                else:
                    # Fallback method was used
                    print(f"   • Retrieval Method: {relevant_data.get('data_retrieval_method', 'Unknown')}")
                    print(f"   • Tool Execution Time: {dr_metrics.get('tool_duration_seconds', 0):.3f}s")
                    print(f"   • Tool Overhead: {dr_metrics.get('overhead_duration_seconds', 0):.3f}s")
                
                # Show which specific tool was used
                if "tool_used" in relevant_data:
                    print(f"   • Primary Tool: {relevant_data['tool_used']}")
                    print(f"   • Tool Description: {relevant_data.get('tool_description', 'N/A')}")
                
                # Performance efficiency
                total_time = dr_metrics.get('total_duration_seconds', 0)
                if total_time > 0:
                    if "agent_duration_seconds" in dr_metrics:
                        efficiency = (dr_metrics.get('agent_duration_seconds', 0) / total_time) * 100
                        print(f"   • Agent Efficiency: {efficiency:.1f}% (agent time vs total time)")
                    elif "tool_duration_seconds" in dr_metrics:
                        efficiency = (dr_metrics.get('tool_duration_seconds', 0) / total_time) * 100
                        print(f"   • Tool Efficiency: {efficiency:.1f}% (tool time vs total time)")
        
        if "messages" in state:
            print(f"\n💬 MESSAGE STATE:")
            messages = state["messages"]
            print(f"   • Total Messages: {len(messages)}")
            print(f"   • Message Types: {', '.join(set([msg.__class__.__name__ for msg in messages]))}")
            print(f"   • Workflow Progress: {len([m for m in messages if 'initiated' in m.content.lower()])} steps initiated")
            
            # Message performance analysis
            if messages:
                # Count messages by type
                message_counts = {}
                for msg in messages:
                    msg_type = msg.__class__.__name__
                    message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
                
                print(f"   • Message Distribution:")
                for msg_type, count in message_counts.items():
                    percentage = (count / len(messages)) * 100
                    print(f"     - {msg_type}: {count} ({percentage:.1f}%)")
                
                # Workflow step analysis
                workflow_steps = [msg.content for msg in messages if 'initiated' in msg.content.lower()]
                if workflow_steps:
                    print(f"   • Workflow Steps: {len(workflow_steps)}")
                    for i, step in enumerate(workflow_steps, 1):
                        print(f"     {i}. {step}")
                
                # Show retriever agent messages
                agent_messages = [msg.content for msg in messages if 'retriever agent' in msg.content.lower()]
                if agent_messages:
                    print(f"   • Retriever Agent Messages: {len(agent_messages)}")
                    for msg in agent_messages:
                        print(f"     - {msg}")
        
        # Performance Summary
        total_duration = sum([
            latency_metrics.get("question_classification", {}).get("total_duration_seconds", 0),
            latency_metrics.get("data_retrieval", {}).get("total_duration_seconds", 0),
            latency_metrics.get("response_generation", {}).get("total_duration_seconds", 0)
        ])
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   • Total Workflow Duration: {total_duration:.3f}s")
        
        # Calculate LLM operations time
        llm_ops_time = sum([
            latency_metrics.get("question_classification", {}).get("llm_duration_seconds", 0),
            latency_metrics.get("response_generation", {}).get("llm_duration_seconds", 0)
        ])
        print(f"   • LLM Operations: {llm_ops_time:.3f}s")
        
        # Calculate agent/tool operations time
        dr_metrics = latency_metrics.get('data_retrieval', {})
        if "agent_duration_seconds" in dr_metrics:
            agent_ops_time = dr_metrics.get('agent_duration_seconds', 0)
            print(f"   • Retriever Agent Operations: {agent_ops_time:.3f}s")
        elif "tool_duration_seconds" in dr_metrics:
            tool_ops_time = dr_metrics.get('tool_duration_seconds', 0)
            print(f"   • Tool Operations: {tool_ops_time:.3f}s")
        else:
            print(f"   • Data Retrieval Operations: {dr_metrics.get('total_duration_seconds', 0):.3f}s")
        
        # Calculate system overhead time
        system_overhead = sum([
            latency_metrics.get("question_classification", {}).get("overhead_duration_seconds", 0),
            latency_metrics.get("data_retrieval", {}).get("overhead_duration_seconds", 0),
            latency_metrics.get("response_generation", {}).get("overhead_duration_seconds", 0)
        ])
        print(f"   • System Overhead: {system_overhead:.3f}s")
        
        # Tool performance analysis
        if tools_used:
            print(f"\n🔧 TOOL PERFORMANCE ANALYSIS:")
            print(f"   • Tools Executed: {len(tools_used)}")
            
            # Determine operation time based on retrieval method
            relevant_data = state.get("relevant_data", {})
            if "retrieval_method" in relevant_data and "retriever_agent" in relevant_data["retrieval_method"]:
                agent_time = dr_metrics.get('agent_duration_seconds', 0)
                print(f"   • Retriever Agent Time: {agent_time:.3f}s")
                print(f"   • Agent Efficiency: {(agent_time/total_duration*100):.1f}% of total workflow time")
                
                # Show agent tool execution details
                if "tool_usage_summary" in relevant_data:
                    tool_summary = relevant_data["tool_usage_summary"]
                    print(f"   • Agent Tool Executions: {sum([info.get('executions', 0) for info in tool_summary.values()])}")
            else:
                tool_time = dr_metrics.get('tool_duration_seconds', 0)
                print(f"   • Tool Execution Time: {tool_time:.3f}s")
                print(f"   • Tool Efficiency: {(tool_time/total_duration*100):.1f}% of total workflow time")
            
            # Tool-specific breakdown if available
            if "retrieval_metadata" in relevant_data:
                metadata = relevant_data["retrieval_metadata"]
                print(f"   • Response Length: {metadata.get('response_length', 0)} characters")
                
                # Calculate processing rate
                if "agent_duration_seconds" in metadata and metadata["agent_duration_seconds"] > 0:
                    rate = metadata.get('response_length', 0) / metadata["agent_duration_seconds"]
                    print(f"   • Agent Processing Rate: {rate:.0f} chars/sec")
                elif "tool_duration_seconds" in metadata and metadata["tool_duration_seconds"] > 0:
                    rate = metadata.get('response_length', 0) / metadata["tool_duration_seconds"]
                    print(f"   • Tool Processing Rate: {rate:.0f} chars/sec")
        
        # Cost analysis with tool usage
        total_input_tokens = sum([
            latency_metrics.get("question_classification", {}).get("input_tokens", 0),
            latency_metrics.get("response_generation", {}).get("input_tokens", 0)
        ])
        total_output_tokens = sum([
            latency_metrics.get("question_classification", {}).get("output_tokens", 0),
            latency_metrics.get("response_generation", {}).get("output_tokens", 0)
        ])
        total_tokens = total_input_tokens + total_output_tokens
        
        print(f"\n🔤 TOKEN USAGE & COST ANALYSIS:")
        print(f"   • Total Input Tokens: {total_input_tokens}")
        print(f"   • Total Output Tokens: {total_output_tokens}")
        print(f"   • Total Tokens: {total_tokens}")
        print(f"   • Estimated Cost (GPT-4o-mini): ${(total_tokens * 0.00001):.4f}")
        
        # Cost per tool operation
        if tools_used:
            cost_per_tool = (total_tokens * 0.00001) / len(tools_used) if tools_used else 0
            print(f"   • Cost per Tool Operation: ${cost_per_tool:.4f}")
        
        # Performance efficiency metrics
        if total_duration > 0:
            print(f"\n⚡ EFFICIENCY METRICS:")
            llm_efficiency = (llm_ops_time / total_duration) * 100
            tool_efficiency = (tool_ops_time / total_duration) * 100
            overhead_efficiency = (system_overhead / total_duration) * 100
            
            print(f"   • LLM Efficiency: {llm_efficiency:.1f}%")
            print(f"   • Tool Efficiency: {tool_efficiency:.1f}%")
            print(f"   • System Overhead: {overhead_efficiency:.1f}%")
            
            # Performance rating
            if overhead_efficiency < 20:
                performance_rating = "Excellent"
            elif overhead_efficiency < 40:
                performance_rating = "Good"
            elif overhead_efficiency < 60:
                performance_rating = "Fair"
            else:
                performance_rating = "Needs Improvement"
            
            print(f"   • Overall Performance Rating: {performance_rating}")
        
        print("="*60)
    
    def _count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text for the specified model"""
        try:
            # Use tiktoken to count tokens
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback to approximate counting if tiktoken fails
            # Rough approximation: 1 token ≈ 4 characters for English text
            return len(text) // 4
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with conditional routing based on classification"""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("retrieve_aws_data", self._retrieve_aws_data_direct)
        workflow.add_node("retrieve_budget_data", self._retrieve_budget_data_direct)
        workflow.add_node("retrieve_vendor_data", self._retrieve_vendor_data_direct)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Set entry point
        workflow.set_entry_point("classify_question")
        
        # Add conditional edges based on classification
        workflow.add_conditional_edges(
            "classify_question",
            self._route_based_on_classification,
            {
                "aws_costs": "retrieve_aws_data",
                "budget": "retrieve_budget_data", 
                "vendor_spend": "retrieve_vendor_data"
            }
        )
        
        # Add edges to generate answer
        workflow.add_edge("retrieve_aws_data", "generate_answer")
        workflow.add_edge("retrieve_budget_data", "generate_answer")
        workflow.add_edge("retrieve_vendor_data", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        # Display workflow information
        print("\n" + "="*60)
        print("🔧 COMPILED WORKFLOW STRUCTURE")
        print("="*60)
        print(f"📊 Workflow Name: {compiled_workflow.__class__.__name__}")
        print(f"🔗 Nodes: {list(compiled_workflow.nodes.keys())}")
        print("="*60)
        
        return compiled_workflow
    
    def _classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the user question into one of three categories with latency tracking"""
        start_time = time.time()
        classification_start = datetime.now()
        
        classification_prompt = f"""
        You are a financial data analyst specializing in company spending analysis. Your task is to classify the following question into exactly one of these three categories:
        
        1. "aws_costs" - Questions about AWS cloud infrastructure costs, EC2, S3, RDS, VPC, CloudWatch, Route 53, or any AWS service expenses
        2. "budget" - Questions about project budgets, team spending, budget variances, budget tracking, or financial planning
        3. "vendor_spend" - Questions about vendor expenses, supplier costs, external services, contractor spending, or third-party service costs
        
        Consider the context carefully. If a question could fit multiple categories, choose the most specific one based on the primary focus of the question.
        
        Question: {state['question']}
        
        Respond with ONLY the category name (aws_costs, budget, or vendor_spend).
        """
        
        try:
            # Track LLM classification time and tokens
            llm_start = time.time()
            
            # Count input tokens
            input_text = f"{self.system_message.content}\n{classification_prompt}"
            input_tokens = self._count_tokens(input_text)
            
            response = self.llm.invoke([
                self.system_message,
                HumanMessage(content=classification_prompt)
            ])
            llm_duration = time.time() - llm_start
            
            # Count output tokens
            output_tokens = self._count_tokens(response.content)
            
            classification = response.content.strip().lower()
            
            # Validate classification
            valid_categories = ["aws_costs", "budget", "vendor_spend"]
            if classification not in valid_categories:
                classification = "budget"  # Default fallback
                
            print(f"Question classified as: {classification}")
            
            total_duration = time.time() - start_time
            classification_end = datetime.now()
            
            # Update latency metrics with token information
            latency_metrics = {
                "question_classification": {
                    "start_time": classification_start.isoformat(),
                    "end_time": classification_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "llm_duration_seconds": round(llm_duration, 3),
                    "overhead_duration_seconds": round(total_duration - llm_duration, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "status": "success"
                }
            }
            
            # Add messages to state for tracking
            messages = state.get("messages", [])
            messages.extend([
                SystemMessage(content="Question classification initiated"),
                HumanMessage(content=classification_prompt),
                response
            ])
            
            return {
                **state,
                "classification": classification,
                "latency_metrics": latency_metrics,
                "messages": messages
            }
            
        except Exception as e:
            total_duration = time.time() - start_time
            classification_end = datetime.now()
            
            # Update latency metrics with error
            latency_metrics = {
                "question_classification": {
                    "start_time": classification_start.isoformat(),
                    "end_time": classification_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "llm_duration_seconds": 0,
                    "overhead_duration_seconds": round(total_duration, 3),
                    "status": "error",
                    "error_message": str(e)
                }
            }
            
            # Add error message to state
            messages = state.get("messages", [])
            messages.append(SystemMessage(content=f"Classification error: {str(e)}"))
            
            return {
                **state,
                "classification": "budget",  # Default fallback
                "error": f"Classification error: {str(e)}",
                "latency_metrics": latency_metrics,
                "messages": messages
            }
    
    def _route_based_on_classification(self, state: WorkflowState) -> str:
        """Route the workflow based on the classification of the question"""
        classification = state.get("classification", "budget") # Default to budget if classification is missing
        return classification
    
    def _retrieve_aws_data_direct(self, state: WorkflowState) -> WorkflowState:
        """Retrieve AWS cost data using the retriever agent with intelligent tool routing"""
        start_time = time.time()
        retrieval_start = datetime.now()
        
        try:
            question = state["question"]
            classification = state["classification"]
            
            # Use the retriever agent for intelligent data retrieval
            agent_start = time.time()
            retrieval_result = self.retriever_agent.retrieve_data(question, classification)
            agent_duration = time.time() - agent_start
            
            if retrieval_result["success"]:
                # Extract data from successful retrieval
                relevant_data = {
                    "data_source": "retriever_agent_aws",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "intelligent_agent_routing",
                    "agent_data": retrieval_result["data"],
                    "tool_execution": retrieval_result.get("tool_execution", []),
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "response_length": len(retrieval_result["data"]),
                        "tools_executed": len(retrieval_result.get("tool_execution", [])),
                        "retrieval_method": "retriever_agent"
                    }
                }
                
                # Get tool usage summary
                tool_summary = self.retriever_agent.get_tool_usage_summary(
                    retrieval_result.get("tool_execution", [])
                )
                relevant_data["tool_usage_summary"] = tool_summary
                
                # Extract tools used from agent execution
                tools_used = list(tool_summary.keys()) if tool_summary else []
                
            else:
                # Fallback to direct tool execution if agent fails
                print(f"Retriever agent failed, falling back to direct tool execution: {retrieval_result.get('error', 'Unknown error')}")
                
                tool_start = time.time()
                aws_data = self.aws_tool._run(question)
                tool_duration = time.time() - tool_start
                
                relevant_data = {
                    "data_source": "aws_data_retrieval_fallback",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "direct_tool_execution_fallback",
                    "raw_data": aws_data,
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "tool_duration_seconds": round(tool_duration, 3),
                        "response_length": len(aws_data),
                        "tools_executed": 1,
                        "retrieval_method": "fallback_direct"
                    }
                }
                
                tools_used = [self.aws_tool.name]
            
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": round(agent_duration, 3),
                    "overhead_duration_seconds": round(total_duration - agent_duration, 3),
                    "status": "success"
                }
            })
            
            # Update tools used and messages
            current_tools_used = state.get("tools_used", [])
            current_tools_used.extend(tools_used)
            
            messages = state.get("messages", [])
            messages.extend([
                SystemMessage(content=f"Retrieved AWS data using retriever agent with tools: {', '.join(tools_used)}"),
                SystemMessage(content=f"Agent execution time: {agent_duration:.3f}s"),
                SystemMessage(content=f"Retrieval method: {relevant_data['data_retrieval_method']}")
            ])
            
            return {
                **state,
                "relevant_data": relevant_data,
                "latency_metrics": latency_metrics,
                "tools_used": current_tools_used,
                "messages": messages
            }
            
        except Exception as e:
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics with error
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": 0,
                    "overhead_duration_seconds": round(total_duration, 3),
                    "status": "error",
                    "error_message": str(e)
                }
            })
            
            # Add error message to state
            messages = state.get("messages", [])
            messages.append(SystemMessage(content=f"AWS data retrieval error: {str(e)}"))
            
            return {
                **state,
                "relevant_data": {"error": f"Error in AWS data retrieval: {str(e)}"},
                "error": str(e),
                "latency_metrics": latency_metrics,
                "messages": messages
            }
    
    def _retrieve_budget_data_direct(self, state: WorkflowState) -> WorkflowState:
        """Retrieve budget tracking data using the retriever agent with intelligent tool routing"""
        start_time = time.time()
        retrieval_start = datetime.now()
        
        try:
            question = state["question"]
            classification = state["classification"]
            
            # Use the retriever agent for intelligent data retrieval
            agent_start = time.time()
            retrieval_result = self.retriever_agent.retrieve_data(question, classification)
            agent_duration = time.time() - agent_start
            
            if retrieval_result["success"]:
                # Extract data from successful retrieval
                relevant_data = {
                    "data_source": "retriever_agent_budget",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "intelligent_agent_routing",
                    "agent_data": retrieval_result["data"],
                    "tool_execution": retrieval_result.get("tool_execution", []),
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "response_length": len(retrieval_result["data"]),
                        "tools_executed": len(retrieval_result.get("tool_execution", [])),
                        "retrieval_method": "retriever_agent"
                    }
                }
                
                # Get tool usage summary
                tool_summary = self.retriever_agent.get_tool_usage_summary(
                    retrieval_result.get("tool_execution", [])
                )
                relevant_data["tool_usage_summary"] = tool_summary
                
                # Extract tools used from agent execution
                tools_used = list(tool_summary.keys()) if tool_summary else []
                
            else:
                # Fallback to direct tool execution if agent fails
                print(f"Retriever agent failed, falling back to direct tool execution: {retrieval_result.get('error', 'Unknown error')}")
                
                tool_start = time.time()
                budget_data = self.budget_tool._run(question)
                tool_duration = time.time() - tool_start
                
                relevant_data = {
                    "data_source": "budget_data_retrieval_fallback",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "direct_tool_execution_fallback",
                    "raw_data": budget_data,
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "tool_duration_seconds": round(tool_duration, 3),
                        "response_length": len(budget_data),
                        "tools_executed": 1,
                        "retrieval_method": "fallback_direct"
                    }
                }
                
                tools_used = [self.budget_tool.name]
            
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": round(agent_duration, 3),
                    "overhead_duration_seconds": round(total_duration - agent_duration, 3),
                    "status": "success"
                }
            })
            
            # Update tools used and messages
            current_tools_used = state.get("tools_used", [])
            current_tools_used.extend(tools_used)
            
            messages = state.get("messages", [])
            messages.extend([
                SystemMessage(content=f"Retrieved budget data using retriever agent with tools: {', '.join(tools_used)}"),
                SystemMessage(content=f"Agent execution time: {agent_duration:.3f}s"),
                SystemMessage(content=f"Retrieval method: {relevant_data['data_retrieval_method']}")
            ])
            
            return {
                **state,
                "relevant_data": relevant_data,
                "latency_metrics": latency_metrics,
                "tools_used": current_tools_used,
                "messages": messages
            }
            
        except Exception as e:
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics with error
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": 0,
                    "overhead_duration_seconds": round(total_duration, 3),
                    "status": "error",
                    "error_message": str(e)
                }
            })
            
            # Add error message to state
            messages = state.get("messages", [])
            messages.append(SystemMessage(content=f"Budget data retrieval error: {str(e)}"))
            
            return {
                **state,
                "relevant_data": {"error": f"Error in budget data retrieval: {str(e)}"},
                "error": str(e),
                "latency_metrics": latency_metrics,
                "messages": messages
            }
    
    def _retrieve_vendor_data_direct(self, state: WorkflowState) -> WorkflowState:
        """Retrieve vendor spending data using the retriever agent with intelligent tool routing"""
        start_time = time.time()
        retrieval_start = datetime.now()
        
        try:
            question = state["question"]
            classification = state["classification"]
            
            # Use the retriever agent for intelligent data retrieval
            agent_start = time.time()
            retrieval_result = self.retriever_agent.retrieve_data(question, classification)
            agent_duration = time.time() - agent_start
            
            if retrieval_result["success"]:
                # Extract data from successful retrieval
                relevant_data = {
                    "data_source": "retriever_agent_vendor",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "intelligent_agent_routing",
                    "agent_data": retrieval_result["data"],
                    "tool_execution": retrieval_result.get("tool_execution", []),
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "response_length": len(retrieval_result["data"]),
                        "tools_executed": len(retrieval_result.get("tool_execution", [])),
                        "retrieval_method": "retriever_agent"
                    }
                }
                
                # Get tool usage summary
                tool_summary = self.retriever_agent.get_tool_usage_summary(
                    retrieval_result.get("tool_execution", [])
                )
                relevant_data["tool_usage_summary"] = tool_summary
                
                # Extract tools used from agent execution
                tools_used = list(tool_summary.keys()) if tool_summary else []
                
            else:
                # Fallback to direct tool execution if agent fails
                print(f"Retriever agent failed, falling back to direct tool execution: {retrieval_result.get('error', 'Unknown error')}")
                
                tool_start = time.time()
                vendor_data = self.vendor_tool._run(question)
                tool_duration = time.time() - tool_start
                
                relevant_data = {
                    "data_source": "vendor_data_retrieval_fallback",
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "direct_tool_execution_fallback",
                    "raw_data": vendor_data,
                    "retrieval_metadata": {
                        "agent_duration_seconds": round(agent_duration, 3),
                        "tool_duration_seconds": round(tool_duration, 3),
                        "response_length": len(vendor_data),
                        "tools_executed": 1,
                        "retrieval_method": "fallback_direct"
                    }
                }
                
                tools_used = [self.vendor_tool.name]
            
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": round(agent_duration, 3),
                    "overhead_duration_seconds": round(total_duration - agent_duration, 3),
                    "status": "success"
                }
            })
            
            # Update tools used and messages
            current_tools_used = state.get("tools_used", [])
            current_tools_used.extend(tools_used)
            
            messages = state.get("messages", [])
            messages.extend([
                SystemMessage(content=f"Retrieved vendor data using retriever agent with tools: {', '.join(tools_used)}"),
                SystemMessage(content=f"Agent execution time: {agent_duration:.3f}s"),
                SystemMessage(content=f"Retrieval method: {relevant_data['data_retrieval_method']}")
            ])
            
            return {
                **state,
                "relevant_data": relevant_data,
                "latency_metrics": latency_metrics,
                "tools_used": current_tools_used,
                "messages": messages
            }
            
        except Exception as e:
            total_duration = time.time() - start_time
            retrieval_end = datetime.now()
            
            # Update latency metrics with error
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "data_retrieval": {
                    "start_time": retrieval_start.isoformat(),
                    "end_time": retrieval_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "agent_duration_seconds": 0,
                    "overhead_duration_seconds": round(total_duration, 3),
                    "status": "error",
                    "error_message": str(e)
                }
            })
            
            # Add error message to state
            messages = state.get("messages", [])
            messages.append(SystemMessage(content=f"Vendor data retrieval error: {str(e)}"))
            
            return {
                **state,
                "relevant_data": {"error": f"Error in vendor data retrieval: {str(e)}"},
                "error": str(e),
                "latency_metrics": latency_metrics,
                "messages": messages
            }
    
    def _retrieve_aws_data(self, question: str) -> Dict[str, Any]:
        """Retrieve AWS cost data with comprehensive analysis capabilities"""
        try:
            df = self.data_loader.get_dataframe("daily-aws-costs")
            
            # Extract key metrics
            total_costs = df[df['Service'] == 'Service total']['Total costs($)'].iloc[0]
            latest_date = df[df['Service'] != 'Service total']['Service'].iloc[-1]
            latest_cost = df[df['Service'] != 'Service total']['Total costs($)'].iloc[-1]
            
            # Get daily costs for trend analysis
            daily_costs = df[df['Service'] != 'Service total'].copy()
            daily_costs['Date'] = pd.to_datetime(daily_costs['Service'])
            daily_costs = daily_costs.sort_values('Date')
            
            # Calculate trend metrics
            avg_daily_cost = daily_costs['Total costs($)'].mean()
            std_daily_cost = daily_costs['Total costs($)'].std()
            min_daily_cost = daily_costs['Total costs($)'].min()
            max_daily_cost = daily_costs['Total costs($)'].max()
            
            # Get top services by cost
            service_costs = df[df['Service'] == 'Service total'].iloc[0].to_dict()
            service_costs.pop('Service', None)
            service_costs.pop('Total costs($)', None)
            
            # Sort services by cost
            sorted_services = sorted(service_costs.items(), key=lambda x: float(x[1]) if x[1] != '' else 0, reverse=True)
            top_services = sorted_services[:5]
            
            # Get service breakdown for anomaly detection
            service_breakdown = []
            for service, cost in sorted_services:
                if cost != '':
                    service_breakdown.append({
                        'service': service,
                        'annual_cost': float(cost),
                        'percentage_of_total': (float(cost) / total_costs) * 100
                    })
            
            # Get recent daily costs for pattern analysis
            recent_daily_costs = daily_costs.tail(30)[['Date', 'Total costs($)']].to_dict('records')
            
            return {
                "dataset": "daily-aws-costs",
                "total_annual_cost": total_costs,
                "latest_date": latest_date,
                "latest_daily_cost": latest_cost,
                "top_services": top_services,
                "service_breakdown": service_breakdown,
                "daily_cost_statistics": {
                    "average_daily_cost": avg_daily_cost,
                    "standard_deviation": std_daily_cost,
                    "minimum_daily_cost": min_daily_cost,
                    "maximum_daily_cost": max_daily_cost,
                    "cost_range": max_daily_cost - min_daily_cost
                },
                "recent_daily_costs": recent_daily_costs,
                "data_shape": df.shape,
                "sample_data": df.head(10).to_dict('records'),
                "full_daily_data": daily_costs[['Date', 'Total costs($)']].to_dict('records')
            }
            
        except Exception as e:
            return {"error": f"Error retrieving AWS data: {str(e)}"}
    
    def _retrieve_budget_data(self, question: str) -> Dict[str, Any]:
        """Retrieve budget tracking data with comprehensive analysis capabilities"""
        try:
            df = self.data_loader.get_dataframe("sample-budget-tracking")
            
            # Calculate budget metrics
            total_budget = df['Monthly Budget'].sum()
            total_spend = df['Actual Spend'].sum()
            total_variance = df['Variance'].sum()
            over_budget_projects = df[df['Variance'] < 0]
            on_track_projects = df[df['Variance'] >= 0]
            
            # Calculate variance percentages
            df['Variance_Percentage'] = (df['Variance'] / df['Monthly Budget']) * 100
            
            # Get projects with highest variance (potential anomalies)
            high_variance_projects = df.nlargest(5, 'Variance_Percentage')[['Team', 'Project', 'Monthly Budget', 'Actual Spend', 'Variance', 'Variance_Percentage']]
            
            # Team performance analysis
            team_performance = df.groupby('Team').agg({
                'Monthly Budget': 'sum',
                'Actual Spend': 'sum',
                'Variance': 'sum',
                'Variance_Percentage': 'mean'
            }).reset_index()
            
            # Status distribution
            status_distribution = df['Status'].value_counts().to_dict()
            
            # Budget utilization analysis
            budget_utilization = (total_spend / total_budget) * 100
            
            return {
                "dataset": "sample-budget-tracking",
                "total_monthly_budget": total_budget,
                "total_actual_spend": total_spend,
                "total_variance": total_variance,
                "budget_utilization_percentage": budget_utilization,
                "over_budget_count": len(over_budget_projects),
                "on_track_count": len(on_track_projects),
                "over_budget_projects": over_budget_projects.to_dict('records'),
                "high_variance_projects": high_variance_projects.to_dict('records'),
                "team_performance": team_performance.to_dict('records'),
                "status_distribution": status_distribution,
                "data_shape": df.shape,
                "sample_data": df.head(10).to_dict('records'),
                "full_budget_data": df.to_dict('records')
            }
            
        except Exception as e:
            return {"error": f"Error retrieving budget data: {str(e)}"}
    
    def _retrieve_vendor_data(self, question: str) -> Dict[str, Any]:
        """Retrieve vendor spending data with comprehensive analysis capabilities"""
        try:
            df = self.data_loader.get_dataframe("sample-vendor-data")
            
            # Calculate vendor metrics
            total_annual_budget = df['Annual Budget Approved'].sum()
            total_current_spend = df['Current Annual Spend'].sum()
            total_variance = df['Budget Variance'].sum()
            
            # Calculate variance percentages
            df['Variance_Percentage'] = (df['Budget Variance'] / df['Annual Budget Approved']) * 100
            
            # Group by vendor type
            vendor_types = df.groupby('Vendor Type').agg({
                'Annual Budget Approved': 'sum',
                'Current Annual Spend': 'sum',
                'Budget Variance': 'sum',
                'Variance_Percentage': 'mean'
            }).reset_index()
            
            # Get high-risk vendors
            high_risk_vendors = df[df['Risk Level'] == 'High']
            
            # Get vendors with highest variance (potential anomalies)
            high_variance_vendors = df.nlargest(5, 'Variance_Percentage')[['Vendor Name', 'Vendor Type', 'Annual Budget Approved', 'Current Annual Spend', 'Budget Variance', 'Variance_Percentage']]
            
            # Risk level distribution
            risk_distribution = df['Risk Level'].value_counts().to_dict()
            
            # Status distribution
            status_distribution = df['Status'].value_counts().to_dict()
            
            # Contract analysis
            contract_analysis = df.groupby('Contract End Date').agg({
                'Vendor Name': 'count',
                'Annual Budget Approved': 'sum'
            }).reset_index()
            contract_analysis.columns = ['Contract_End_Date', 'Vendor_Count', 'Total_Budget']
            
            # Budget utilization analysis
            budget_utilization = (total_current_spend / total_annual_budget) * 100
            
            return {
                "dataset": "sample-vendor-data",
                "total_annual_budget": total_annual_budget,
                "total_current_spend": total_current_spend,
                "total_variance": total_variance,
                "budget_utilization_percentage": budget_utilization,
                "vendor_types_summary": vendor_types.to_dict('records'),
                "high_risk_vendors": high_risk_vendors.to_dict('records'),
                "high_variance_vendors": high_variance_vendors.to_dict('records'),
                "risk_distribution": risk_distribution,
                "status_distribution": status_distribution,
                "contract_analysis": contract_analysis.to_dict('records'),
                "data_shape": df.shape,
                "sample_data": df.head(10).to_dict('records'),
                "full_vendor_data": df.to_dict('records')
            }
            
        except Exception as e:
            return {"error": f"Error retrieving vendor data: {str(e)}"}
    
    def _generate_answer(self, state: WorkflowState) -> WorkflowState:
        """Generate a comprehensive answer based on retrieved data with latency tracking"""
        start_time = time.time()
        generation_start = datetime.now()
        
        try:
            question = state['question']
            classification = state['classification']
            relevant_data = state['relevant_data']
            tools_used = state.get('tools_used', [])
            
            if 'error' in relevant_data:
                answer = f"Error retrieving data: {relevant_data['error']}"
                total_duration = time.time() - start_time
                generation_end = datetime.now()
                
                # Update latency metrics for error case
                latency_metrics = state.get("latency_metrics", {})
                latency_metrics.update({
                    "response_generation": {
                        "start_time": generation_start.isoformat(),
                        "end_time": generation_end.isoformat(),
                        "total_duration_seconds": round(total_duration, 3),
                        "llm_duration_seconds": 0,
                        "overhead_duration_seconds": round(total_duration, 3),
                        "response_length": len(answer),
                        "status": "error_skip_llm"
                    }
                })
                
                # Add error message to state
                messages = state.get("messages", [])
                messages.append(SystemMessage(content=f"Generated error answer: {answer}"))
                
                return {
                    **state,
                    "final_answer": answer,
                    "latency_metrics": latency_metrics,
                    "messages": messages
                }
            else:
                # Create a detailed prompt for answer generation with full data access
                answer_prompt = f"""
                You are a senior financial analyst at a company. Analyze the data and answer the following question with a professional, executive-ready response. 
                Your analysis must emphasize detecting, quantifying, and explaining anomalies (unusual spikes, drops, or outliers) along with their business implications. 

                Question: {question}
                Classification: {classification}
                Tools Used: {', '.join(tools_used)}

                **COMPLETE DATA ACCESS FOR ANALYSIS:**
                {self._format_complete_data_for_analysis(relevant_data)}

                **ANALYSIS REQUIREMENTS:**
                Please provide a comprehensive, professional answer that follows this structure:

                **EXECUTIVE SUMMARY**
                - 1–2 sentence overview of the key findings
                - Explicitly mention if any anomalies were detected, their severity, and their overall significance

                **DETAILED ANALYSIS**
                - Present the specific data and metrics that answer the question
                - Use exact numbers and percentages from the data
                - **ANOMALY DETECTION**:
                - Highlight unusual spikes, drops, or deviations from budget/historical baselines
                - For each anomaly, provide:
                    • Period / Vendor / Department (as applicable)  
                    • Metric affected (e.g., Current Annual Spend, Budget Variance)  
                    • Actual Value vs. Expected Value (or prior period)  
                    • % Deviation from baseline  
                    • **Severity Score**: High / Medium / Low, based on business impact  
                - Separate **positive anomalies** (favorable variances) from **negative anomalies** (unfavorable variances)
                - Highlight recurring trends, patterns, or data quality issues
                - Compare current vs. historical/budgeted data to contextualize anomalies

                **KEY INSIGHTS**
                - 2–3 concise bullet points with the most important takeaways
                - **ANOMALY INSIGHTS**:
                - List the anomalies with severity scores and explain business implications
                - Call out the most critical anomaly and its potential root cause

                **RECOMMENDATIONS**
                - Actionable next steps tied directly to findings
                - Risk mitigation strategies for negative anomalies
                - Optimization opportunities from positive anomalies
                - **ANOMALY RESPONSE**:
                - For each anomaly (especially High severity), suggest specific actions to investigate, resolve, or capitalize on the finding

                **ANOMALY DATA RETURN**
                Return a structured anomaly table with the following columns:
                - Row/Period Identifier
                - Metric
                - Actual Value
                - Expected Value / Benchmark
                - % Deviation
                - Severity Score
                - Business Impact (1–2 sentence explanation)

                **EXAMPLES**

                Example 1 — Executive Summary:
                - "Q2 AWS cloud spend exceeded the approved budget by 65%, representing a High severity anomaly. While other vendors remained within budget, this deviation poses significant cost overrun risk."

                Example 2 — Anomaly Data Return Table:

                | Period   | Metric               | Actual Value | Expected Value | % Deviation | Severity Score | Business Impact |
                |----------|----------------------|--------------|----------------|-------------|----------------|-----------------|
                | Q2-2025  | AWS Current Spend    | $850,000     | $515,000       | +65%        | High           | Significant overspend vs. budget; may impact EBITDA if not corrected. |
                | Q1-2025  | Salesforce Variance  | +$25,000     | $0             | +12%        | Low            | Slight overage within tolerance, but trend should be monitored. |
                - GCP spend dropped by 50% in Q4-2024, suggesting underutilization (Medium severity).  
                - Salesforce spend slightly over budget (+12%), not critical but worth monitoring (Low severity).  

                Example 4 — Recommendations:
                - Immediately review AWS workloads to identify drivers of overspend and implement cost-control measures.  
                - Reassess GCP commitments to optimize reserved instance utilization.  
                - Track Salesforce license allocations to prevent creeping overspend.  

                **DATA SOURCES & QUALITY**
                - Reference the datasets used for this analysis
                - Note any missing, incomplete, or inconsistent data
                - Confidence level in the analysis based on completeness and reliability of the dataset

                **IMPORTANT INSTRUCTIONS:**
                - Analyze the complete dataset provided above
                - Always return anomalies with their details and severity scores
                - Use quantitative evidence (numbers and % variances) to justify anomalies
                - Clearly call out the business impact of each anomaly
                - Do not fabricate data or reference anything outside the provided dataset
                - Format your response with professional, executive-ready structure, clear headings, and bullet points

                Answer:

                """
                
                # Track LLM generation time and tokens
                llm_start = time.time()
                
                # Count input tokens
                input_text = f"{self.system_message.content}\n{answer_prompt}"
                input_tokens = self._count_tokens(input_text)
                
                response = self.llm.invoke([
                    self.system_message,
                    HumanMessage(content=answer_prompt)
                ])
                llm_duration = time.time() - llm_start
                
                # Count output tokens
                output_tokens = self._count_tokens(response.content)
                
                answer = response.content.strip()
                total_duration = time.time() - start_time
                generation_end = datetime.now()
                
                # Update latency metrics with token information
                latency_metrics = state.get("latency_metrics", {})
                latency_metrics.update({
                    "response_generation": {
                        "start_time": generation_start.isoformat(),
                        "end_time": generation_end.isoformat(),
                        "total_duration_seconds": round(total_duration, 3),
                        "llm_duration_seconds": round(llm_duration, 3),
                        "overhead_duration_seconds": round(total_duration - llm_duration, 3),
                        "response_length": len(answer),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "status": "success"
                    }
                })
                
                # Add final answer message to state
                messages = state.get("messages", [])
                messages.extend([
                    SystemMessage(content="Generated final answer using LLM"),
                    SystemMessage(content=f"Answer length: {len(answer)} characters"),
                    SystemMessage(content=f"Tools used in workflow: {', '.join(tools_used)}")
                ])
                
                return {
                    **state,
                    "final_answer": answer,
                    "latency_metrics": latency_metrics,
                    "messages": messages
                }
            
        except Exception as e:
            total_duration = time.time() - start_time
            generation_end = datetime.now()
            
            # Update latency metrics with error
            latency_metrics = state.get("latency_metrics", {})
            latency_metrics.update({
                "response_generation": {
                    "start_time": generation_start.isoformat(),
                    "end_time": generation_end.isoformat(),
                    "total_duration_seconds": round(total_duration, 3),
                    "llm_duration_seconds": 0,
                    "overhead_duration_seconds": round(total_duration, 3),
                    "response_length": 0,
                    "status": "error",
                    "error_message": str(e)
                }
            })
            
            # Add error message to state
            messages = state.get("messages", [])
            messages.append(SystemMessage(content=f"Answer generation error: {str(e)}"))
            
            return {
                **state,
                "final_answer": f"Error generating answer: {str(e)}",
                "error": str(e),
                "latency_metrics": latency_metrics,
                "messages": messages
            }
    
    def _format_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format data for the LLM prompt"""
        if 'error' in data:
            return f"Error: {data['error']}"
        
        formatted = []
        
        # Add dataset information
        if 'dataset' in data:
            formatted.append(f"Dataset: {data['dataset']}")
            formatted.append("")
        
        # Add key metrics and insights
        for key, value in data.items():
            if key not in ['dataset', 'data_shape', 'sample_data']:
                if isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        formatted.append(f"{key}: {len(value)} items")
                        # Show first few items for context
                        if len(value) <= 3:
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    item_summary = {k: v for k, v in list(item.items())[:3]}
                                    formatted.append(f"  - Item {i+1}: {item_summary}")
                        else:
                            formatted.append(f"  - Sample items available (showing first 3)")
                else:
                    # Format numbers nicely
                    if isinstance(value, (int, float)):
                        if key.lower().find('cost') != -1 or key.lower().find('budget') != -1 or key.lower().find('spend') != -1:
                            formatted.append(f"{key}: ${value:,.2f}")
                        elif key.lower().find('percentage') != -1 or key.lower().find('variance') != -1:
                            formatted.append(f"{key}: {value:.2f}%")
                        else:
                            formatted.append(f"{key}: {value:,}")
                    else:
                        formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def _format_complete_data_for_analysis(self, data: Dict[str, Any]) -> str:
        """Format complete data structure for comprehensive analysis and anomaly detection"""
        if 'error' in data:
            return f"Error: {data['error']}"
        
        formatted = []
        
        # Add dataset information
        if 'dataset' in data:
            formatted.append(f"**DATASET: {data['dataset']}**")
            formatted.append("")
        
        # Add data shape for context
        if 'data_shape' in data:
            formatted.append(f"**DATA STRUCTURE:** {data['data_shape'][0]} rows × {data['data_shape'][1]} columns")
            formatted.append("")
        
        # Add complete data for analysis
        for key, value in data.items():
            if key not in ['dataset', 'data_shape']:
                formatted.append(f"**{key.upper().replace('_', ' ')}:**")
                
                if isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        formatted.append(f"Total items: {len(value)}")
                        formatted.append("")
                        
                        # Show all items for complete analysis
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                formatted.append(f"  **Item {i+1}:**")
                                for k, v in item.items():
                                    if isinstance(v, (int, float)):
                                        if k.lower().find('cost') != -1 or k.lower().find('budget') != -1 or k.lower().find('spend') != -1:
                                            formatted.append(f"    {k}: ${v:,.2f}")
                                        elif k.lower().find('percentage') != -1 or k.lower().find('variance') != -1:
                                            formatted.append(f"    {k}: {v:.2f}%")
                                        else:
                                            formatted.append(f"    {k}: {v:,}")
                                    else:
                                        formatted.append(f"    {k}: {v}")
                                formatted.append("")
                        else:
                            formatted.append("")
                else:
                    # Format numbers nicely
                    if isinstance(value, (int, float)):
                        if key.lower().find('cost') != -1 or key.lower().find('budget') != -1 or key.lower().find('spend') != -1:
                            formatted.append(f"${value:,.2f}")
                        elif key.lower().find('percentage') != -1 or key.lower().find('variance') != -1:
                            formatted.append(f"{value:.2f}%")
                        else:
                            formatted.append(f"{value:,}")
                    else:
                        formatted.append(f"{value}")
                    formatted.append("")
        
        # Add sample data for additional context
        if 'sample_data' in data and data['sample_data']:
            formatted.append("**SAMPLE DATA FOR CONTEXT:**")
            formatted.append("")
            for i, row in enumerate(data['sample_data'][:5]):  # Show first 5 rows
                formatted.append(f"  **Row {i+1}:**")
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        if k.lower().find('cost') != -1 or k.lower().find('budget') != -1 or k.lower().find('spend') != -1:
                            formatted.append(f"    {k}: ${v:,.2f}")
                        elif k.lower().find('percentage') != -1 or k.lower().find('variance') != -1:
                            formatted.append(f"    {k}: {v:.2f}%")
                        else:
                            formatted.append(f"    {k}: {v:,}")
                    else:
                        formatted.append(f"    {k}: {v}")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the complete workflow"""
        
        initial_state = {
            "question": question,
            "classification": "",
            "relevant_data": {},
            "final_answer": "",
            "error": "",
            "messages": [], # Initialize messages
            "tools_used": [] # Initialize tools_used
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "error": f"Workflow execution error: {str(e)}"
            }

    def display_workflow(self):
        """Display the enhanced workflow structure with retriever agent and tool routing"""
        print("\n" + "="*100)
        print("🚀 ENHANCED AUTO-SPEND WORKFLOW WITH RETRIEVER AGENT")
        print("="*100)
        
        # Workflow Overview
        print("\n📋 WORKFLOW OVERVIEW")
        print("   This is an intelligent financial analysis workflow that uses:")
        print("   • Advanced question classification")
        print("   • Conditional routing based on classification")
        print("   • Retriever agent for intelligent tool selection")
        print("   • Specialized data retrieval tools")
        print("   • Comprehensive performance monitoring")
        
        # Workflow Structure
        print("\n🔄 WORKFLOW STRUCTURE")
        print("   START")
        print("     ↓")
        print("   📝 classify_question")
        print("     ↓")
        print("   🔄 conditional_routing")
        print("     ↓")
        print("   🤖 retriever_agent_execution")
        print("     ↓")
        print("   🔧 tool_execution")
        print("     ↓")
        print("   💡 generate_answer")
        print("     ↓")
        print("   END")
        
        # Conditional Routing Details
        print("\n🔄 CONDITIONAL ROUTING PATHS")
        print("   Based on question classification, the workflow routes to specific data retrieval nodes:")
        print("   ")
        print("   🔴 aws_costs → retrieve_aws_data")
        print("      ├─ Tool: AWS Data Retrieval")
        print("      ├─ Data: Cloud infrastructure costs, service breakdowns, trends")
        print("      └─ Features: Anomaly detection, cost optimization insights")
        print("   ")
        print("   🟡 budget → retrieve_budget_data")
        print("      ├─ Tool: Budget Data Retrieval")
        print("      ├─ Data: Project budgets, team spending, variance analysis")
        print("      └─ Features: Budget tracking, variance reporting, project monitoring")
        print("   ")
        print("   🟢 vendor_spend → retrieve_vendor_data")
        print("      ├─ Tool: Vendor Data Retrieval")
        print("      ├─ Data: Contract details, risk assessment, spending patterns")
        print("      └─ Features: Risk analysis, contract monitoring, vendor optimization")
        
        # Retriever Agent Details
        print("\n🤖 RETRIEVER AGENT ARCHITECTURE")
        print("   The retriever agent intelligently routes classification responses to specific tools:")
        print("   ")
        print("   🧠 Intelligent Routing:")
        print("      • Analyzes question context and classification")
        print("      • Determines optimal tool selection")
        print("      • Can use multiple tools for complex questions")
        print("   ")
        print("   🔄 Fallback Strategy:")
        print("      • Automatically falls back to direct tool execution if agent fails")
        print("      • Ensures workflow reliability and continuity")
        print("      • Maintains performance even during agent issues")
        print("   ")
        print("   📊 Tool Usage Tracking:")
        print("      • Records which tools were executed")
        print("      • Tracks execution counts and performance")
        print("      • Provides analytics for optimization")
        
        # Data Retrieval Tools
        print("\n🔧 DATA RETRIEVAL TOOLS")
        print("   Each tool is specialized for specific financial data analysis:")
        print("   ")
        print("   ☁️  AWS Data Retrieval Tool:")
        print("      • Name: aws_data_retrieval")
        print("      • Purpose: Cloud infrastructure cost analysis")
        print("      • Capabilities: Daily costs, service breakdowns, trend analysis")
        print("      • Output: Comprehensive AWS cost insights with anomaly detection")
        print("   ")
        print("   💰 Budget Data Retrieval Tool:")
        print("      • Name: budget_data_retrieval")
        print("      • Purpose: Project budget tracking and variance analysis")
        print("      • Capabilities: Budget monitoring, team spending, status tracking")
        print("      • Output: Budget performance insights with variance reporting")
        print("   ")
        print("   🏢 Vendor Data Retrieval Tool:")
        print("      • Name: vendor_data_retrieval")
        print("      • Purpose: Vendor spending and contract analysis")
        print("      • Capabilities: Contract monitoring, risk assessment, spending patterns")
        print("      • Output: Vendor performance insights with risk analysis")
        
        # State Management
        print("\n💬 STATE MANAGEMENT & TRACKING")
        print("   The workflow maintains comprehensive state throughout execution:")
        print("   ")
        print("   📝 Message State:")
        print("      • Tracks all workflow progression messages")
        print("      • Records classification, tool execution, and answer generation")
        print("      • Provides complete audit trail for debugging")
        print("   ")
        print("   🔧 Tool Usage Tracking:")
        print("      • Records which tools were executed")
        print("      • Tracks execution counts and performance metrics")
        print("      • Enables optimization and cost analysis")
        print("   ")
        print("   ⚡ Latency Metrics:")
        print("      • Comprehensive performance monitoring")
        print("      • Step-by-step timing analysis")
        print("      • Efficiency calculations and ratings")
        
        # Performance Monitoring
        print("\n⚡ ENHANCED PERFORMANCE MONITORING")
        print("   The workflow provides detailed performance analytics:")
        print("   ")
        print("   📊 Classification Metrics:")
        print("      • Question analysis time and token usage")
        print("      • Classification accuracy and confidence")
        print("      • LLM performance and cost tracking")
        print("   ")
        print("   🤖 Retriever Agent Metrics:")
        print("      • Agent execution time and efficiency")
        print("      • Tool routing decisions and success rates")
        print("      • Fallback frequency and performance")
        print("   ")
        print("   🔧 Tool Execution Metrics:")
        print("      • Individual tool performance times")
        print("      • Data retrieval efficiency and rates")
        print("      • Tool usage analytics and optimization")
        print("   ")
        print("   💡 Answer Generation Metrics:")
        print("      • LLM processing time and quality")
        print("      • Response generation efficiency")
        print("      • Token cost analysis and optimization")
        
        # Workflow Benefits
        print("\n🎯 WORKFLOW BENEFITS")
        print("   This enhanced architecture provides several key advantages:")
        print("   ")
        print("   🚀 Efficiency:")
        print("      • Intelligent tool routing reduces unnecessary executions")
        print("      • Conditional routing optimizes workflow paths")
        print("      • Fallback mechanisms ensure reliability")
        print("   ")
        print("   🧠 Intelligence:")
        print("      • Context-aware tool selection")
        print("      • Adaptive routing based on question complexity")
        print("      • Multi-tool integration for comprehensive analysis")
        print("   ")
        print("   📊 Transparency:")
        print("      • Complete visibility into tool usage")
        print("      • Detailed performance metrics and analytics")
        print("      • Full audit trail for compliance and debugging")
        print("   ")
        print("   🔧 Maintainability:")
        print("      • Clear separation of concerns")
        print("      • Modular tool architecture")
        print("      • Easy addition of new tools and capabilities")
        
        # Usage Examples
        print("\n💡 USAGE EXAMPLES")
        print("   The workflow handles various types of financial analysis questions:")
        print("   ")
        print("   🔴 AWS Cost Analysis:")
        print("      • Question: 'What are our AWS costs for this month?'")
        print("      • Classification: aws_costs")
        print("      • Route: classify_question → retrieve_aws_data")
        print("      • Output: Comprehensive cloud cost analysis with anomalies")
        print("   ")
        print("   🟡 Budget Tracking:")
        print("      • Question: 'How is our budget tracking across projects?'")
        print("      • Classification: budget")
        print("      • Route: classify_question → retrieve_budget_data")
        print("      • Output: Project budget performance with variance analysis")
        print("   ")
        print("   🟢 Vendor Spending:")
        print("      • Question: 'What are our vendor spending patterns?'")
        print("      • Classification: vendor_spend")
        print("      • Route: classify_question → retrieve_vendor_data")
        print("      • Output: Vendor performance insights with risk assessment")
        print("   ")
        print("   🌟 Complex Analysis:")
        print("      • Question: 'Show me our cloud costs and budget variances'")
        print("      • Classification: aws_costs (primary) + budget (secondary)")
        print("      • Route: Multiple tool execution via retriever agent")
        print("      • Output: Integrated analysis across multiple data sources")
        
        print("\n" + "="*100)
        print("🎉 Enhanced Auto-Spend Workflow Display Complete!")
        print("   Use workflow.visualize_workflow() to see the interactive HTML diagram")
        print("   Use workflow.display_latency_metrics() to see performance analytics")
        print("="*100)
