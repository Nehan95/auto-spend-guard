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

# Load environment variables
load_dotenv()

# Define the state structure
class WorkflowState(TypedDict):
    question: str
    classification: str
    relevant_data: Dict[str, Any]
    final_answer: str
    error: str

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
            model="gpt-3.5-turbo",
            temperature=0,
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
        
        # Create the data retrieval agent
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
        print(f"   • Flow: classify_question → retrieve_data (agent-based) → generate_answer → END")
        print(f"   • State Management: WorkflowState with question, classification, relevant_data, final_answer")
        print(f"   • Data Retrieval: Intelligent agent with specialized tools")
        print(f"   • Debug Mode: {self.workflow.debug}")
        
        # Agent information
        print(f"\n🤖 DATA RETRIEVAL AGENT:")
        print(f"   • Agent Type: OpenAI Functions Agent")
        print(f"   • Tools: {len([self.aws_tool, self.budget_tool, self.vendor_tool])} specialized tools")
        print(f"   • Capabilities: Intelligent data selection, anomaly detection, trend analysis")
        
        print("="*60)
    
    def visualize_workflow(self, save_path: str = "workflow_graph.html"):
        """Create a visual representation of the workflow graph using HTML"""
        try:
            from workflow_visualizer import WorkflowVisualizer
            
            # Use the visualizer module
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
        """Display a text-based representation of the workflow"""
        try:
            from workflow_visualizer import WorkflowVisualizer
            WorkflowVisualizer.display_text_workflow()
        except ImportError:
            # Fallback to inline text display if module not available
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
            print("="*60)
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("retrieve_data", self._retrieve_data)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Set entry point
        workflow.set_entry_point("classify_question")
        
        # Add edges
        workflow.add_edge("classify_question", "retrieve_data")
        workflow.add_edge("retrieve_data", "generate_answer")
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
        """Classify the user question into one of three categories"""
        
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
            response = self.llm.invoke([
                self.system_message,
                HumanMessage(content=classification_prompt)
            ])
            classification = response.content.strip().lower()
            
            # Validate classification
            valid_categories = ["aws_costs", "budget", "vendor_spend"]
            if classification not in valid_categories:
                classification = "budget"  # Default fallback
                
            print(f"Question classified as: {classification}")
            
            return {
                **state,
                "classification": classification
            }
            
        except Exception as e:
            return {
                **state,
                "classification": "budget",  # Default fallback
                "error": f"Classification error: {str(e)}"
            }
    
    def _retrieve_data(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant data using the intelligent agent"""
        try:
            question = state["question"]
            classification = state["classification"]
            
            # Create a context-aware query for the agent
            agent_query = f"""
Based on the classification '{classification}', please analyze the following question and retrieve relevant data:

Question: {question}

Please provide a comprehensive analysis including:
1. Key metrics and statistics
2. Anomaly detection
3. Trends and patterns
4. Business insights
5. Recommendations

Use the most appropriate data retrieval tools to gather comprehensive information.
            """
            
            # Use the agent to retrieve and analyze data
            agent_response = self.data_retrieval_agent.invoke({
                "input": agent_query
            })
            
            # Extract the agent's response
            if "output" in agent_response:
                relevant_data = {
                    "agent_analysis": agent_response["output"],
                    "classification": classification,
                    "query": question,
                    "data_retrieval_method": "intelligent_agent",
                    "tools_used": [tool.name for tool in [self.aws_tool, self.budget_tool, self.vendor_tool]]
                }
            else:
                relevant_data = {
                    "error": "Agent response format unexpected",
                    "raw_response": agent_response
                }
            
            return {
                **state,
                "relevant_data": relevant_data
            }
            
        except Exception as e:
            return {
                **state,
                "relevant_data": {"error": f"Error in agent-based data retrieval: {str(e)}"},
                "error": str(e)
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
        """Generate a comprehensive answer based on retrieved data"""
        
        try:
            question = state['question']
            classification = state['classification']
            relevant_data = state['relevant_data']
            
            if 'error' in relevant_data:
                answer = f"Error retrieving data: {relevant_data['error']}"
            else:
                # Create a detailed prompt for answer generation with full data access
                answer_prompt = f"""
                You are a senior financial analyst at a company. Analyze the data and answer the following question with a professional, executive-ready response. 
                Your analysis must emphasize detecting, quantifying, and explaining anomalies (unusual spikes, drops, or outliers) along with their business implications. 

                Question: {question}
                Classification: {classification}

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
                | Q4-2024  | GCP Current Spend    | $120,000     | $240,000       | -50%        | Medium         | Underutilization of reserved instances; potential efficiency issue. |

                Example 3 — Key Insights:
                - AWS costs surged in Q2-2025, exceeding budget by 65% (High severity anomaly).  
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
                
                response = self.llm.invoke([
                    self.system_message,
                    HumanMessage(content=answer_prompt)
                ])
                answer = response.content.strip()
            
            return {
                **state,
                "final_answer": answer
            }
            
        except Exception as e:
            return {
                **state,
                "final_answer": f"Error generating answer: {str(e)}",
                "error": str(e)
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
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "error": f"Workflow execution error: {str(e)}"
            }
