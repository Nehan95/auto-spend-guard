from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
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
        
        return workflow.compile()
    
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
        """Retrieve relevant data based on the classification"""
        
        try:
            classification = state['classification']
            question = state['question']
            
            if classification == "aws_costs":
                relevant_data = self._retrieve_aws_data(question)
            elif classification == "budget":
                relevant_data = self._retrieve_budget_data(question)
            elif classification == "vendor_spend":
                relevant_data = self._retrieve_vendor_data(question)
            else:
                relevant_data = {"error": "Unknown classification"}
            
            return {
                **state,
                "relevant_data": relevant_data
            }
            
        except Exception as e:
            return {
                **state,
                "relevant_data": {"error": f"Data retrieval error: {str(e)}"},
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
                You are a senior financial analyst at a company. Analyze the data and answer the following question with a professional, executive-ready response. Report any unusual spikes or drops in the data.
                
                Question: {question}
                Classification: {classification}
                
                **COMPLETE DATA ACCESS FOR ANALYSIS:**
                
                {self._format_complete_data_for_analysis(relevant_data)}
                
                **ANALYSIS REQUIREMENTS:**
                
                Please provide a comprehensive, professional answer that follows this structure:
                
                **EXECUTIVE SUMMARY**
                - Brief 1-2 sentence overview of the key findings
                
                **DETAILED ANALYSIS**
                - Present the specific data and metrics that answer the question
                - Use exact numbers and percentages from the data
                - **ANOMALY DETECTION**: Identify and explain any unusual spikes, drops, or patterns in the data
                - Highlight trends, patterns, and data quality issues
                - Compare current vs. historical data where available
                
                **KEY INSIGHTS**
                - 2-3 bullet points of the most important takeaways
                - Business implications of the findings
                - **ANOMALY IMPACT**: Business impact of any detected anomalies
                
                **RECOMMENDATIONS** (if applicable)
                - Actionable next steps
                - Risk mitigation strategies
                - Optimization opportunities
                - **ANOMALY RESPONSE**: Specific actions to address any detected anomalies
                
                **DATA SOURCES & QUALITY**
                - Reference the datasets used for this analysis
                - Note any data quality issues or missing information
                - Confidence level in the analysis based on data completeness
                
                **IMPORTANT**: Analyze the complete data structure provided above to identify any anomalies, outliers, or unusual patterns. Use statistical analysis where appropriate to detect significant deviations from normal patterns.
                Do not make up any data or use any data that is not provided in the data structure.
                Format your response professionally with clear headings, bullet points, and proper spacing. Use business language appropriate for executive stakeholders.
                
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
