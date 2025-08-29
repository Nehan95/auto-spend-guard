#!/usr/bin/env python3
"""
Streamlit App for Auto-Spend Guard Workflow
A user-friendly web interface for the LangGraph-based spend analysis workflow
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from langgraph_workflow import SpendAnalyzerWorkflow
    from dataloader import DataLoader
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Spend Guardian",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .workflow-step {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .success-step {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .error-step {
        background: #ffebee;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def display_workflow_visualization():
    """Display the workflow visualization HTML"""
    st.header("🔧 Spend Guardian Workflow Visualization")
    
    # Read the workflow visualization HTML file
    workflow_html_path = Path(__file__).parent / "workflow_visualization.html"
    
    if workflow_html_path.exists():
        with open(workflow_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display the HTML content
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Add download button for the HTML file
        st.download_button(
            label="📥 Download Spend Guardian Workflow",
            data=html_content,
            file_name="spend_guardian_workflow.html",
            mime="text/html",
            help="Download the Spend Guardian workflow visualization as an HTML file"
        )
    else:
        st.error("Workflow visualization file not found. Please ensure 'workflow_visualization.html' exists in the same directory.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💰 Spend Guardian</h1>
        <p>AI-Powered Financial Analysis & Anomaly Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        selected_model = st.selectbox(
            "Select AI Model",
            model_options,
            index=0,
            help="Choose the OpenAI model for analysis"
        )
        
        # Temperature setting
        temperature = st.slider(
            "AI Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
        
        # Analysis type
        analysis_types = [
            "General Analysis",
            "Anomaly Detection",
            "Cost Optimization",
            "Budget Tracking",
            "Vendor Analysis"
        ]
        selected_analysis = st.selectbox(
            "Analysis Type",
            analysis_types,
            index=0
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            enable_debug = st.checkbox("Enable Debug Mode", value=False)
            max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=2000)
        
        st.markdown("---")
        
        # Workflow Visualization Button
        if st.button("🔧 View Workflow", use_container_width=True, type="secondary"):
            st.session_state.show_workflow = True
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("• AWS Cost Data")
        st.markdown("• Budget Tracking")
        st.markdown("• Vendor Information")
        
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("• 🔍 Intelligent Analysis")
        st.markdown("• 📊 Anomaly Detection")
        st.markdown("• 💡 Cost Optimization")
        st.markdown("• 📈 Trend Analysis")
        
        st.markdown("---")
        st.markdown("**About Spend Guardian:**")
        st.markdown("• 🛡️ Protects your budget")
        st.markdown("• 🔍 Monitors spending patterns")
        st.markdown("• ⚠️ Detects anomalies early")
        st.markdown("• 💰 Optimizes costs automatically")
    
    # Check if workflow visualization should be displayed
    if st.session_state.get('show_workflow', False):
        display_workflow_visualization()
        # Reset the flag after displaying
        st.session_state.show_workflow = False
        return
    
    # Main content area
    st.header("📝 Analysis")
    
    # Query input
    query_placeholder = "e.g., 'Analyze our AWS spending patterns and identify cost optimization opportunities'"
    user_query = st.text_area(
        "Enter your financial analysis question:",
        placeholder=query_placeholder,
        height=120,
        help="Ask Spend Guardian about spending patterns, anomalies, cost optimization, or budget analysis"
    )
    
    # Initialize user_query if it's None
    if user_query is None:
        user_query = ""
    
    # Store query in session state
    if 'user_query' in st.session_state:
        user_query = st.session_state.user_query
        st.session_state.user_query = None
    
    # Analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if user_query and user_query.strip():
            run_analysis(user_query, selected_model, temperature, enable_debug)
        else:
            st.warning("Please enter a query for Spend Guardian to analyze.")
    
    # Display recent analyses
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.header("📚 Recent Analyses")
        
        for i, analysis in enumerate(st.session_state.analysis_history[-3:]):
            with st.expander(f"Analysis {i+1}: {analysis['query'][:50]}..."):
                st.markdown(f"**Query:** {analysis['query']}")
                st.markdown(f"**Timestamp:** {analysis['timestamp']}")
                st.markdown(f"**Status:** {analysis['status']}")
                
                if analysis['status'] == 'completed':
                    st.markdown("**Summary:**")
                    st.markdown(analysis['summary'][:200] + "...")

def run_analysis(query, model, temperature, enable_debug):
    """Run the analysis workflow"""
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize workflow
        status_text.text("Initializing workflow...")
        progress_bar.progress(20)
        
        # Create workflow with custom model and temperature
        workflow = SpendAnalyzerWorkflow()
        
        # Update the LLM configuration by creating a new instance
        from langchain_openai import ChatOpenAI
        workflow.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Step 2: Run analysis
        status_text.text("Running analysis...")
        progress_bar.progress(50)
        
        result = workflow.run(query)
        
        # Step 3: Process results
        status_text.text("Processing results...")
        progress_bar.progress(80)
        
        # Step 4: Display results
        status_text.text("Displaying results...")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(result, query, workflow, enable_debug)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")

def display_results(result, query, workflow, enable_debug):
    """Display the analysis results"""
    
    st.header("📊 Analysis Results")
    
    # Display the main answer
    if result.get('final_answer'):
        st.markdown("### 💡 Analysis Summary")
        st.markdown(result['final_answer'])
    else:
        st.warning("No analysis results generated.")
    
    # Display workflow details
    if enable_debug:
        with st.expander("🔧 Workflow Details"):
            st.markdown(f"**Question Classification:** {result.get('classification', 'N/A')}")
            
            if result.get('relevant_data'):
                st.markdown("**Retrieved Data:**")
                for key, value in result['relevant_data'].items():
                    if isinstance(value, str) and len(value) > 100:
                        st.markdown(f"**{key}:** {value[:100]}...")
                    else:
                        st.markdown(f"**{key}:** {value}")
    
    # Display comprehensive latency and token metrics as expandable dropdown
    if result.get('latency_metrics'):
        with st.expander("⚡ Performance & Token Metrics", expanded=False):
            # Create tabs for different metric views
            tab1, tab2, tab3 = st.tabs(["📊 Node Performance", "🔢 Token Usage", "📈 Summary Metrics"])
            
            with tab1:
                st.subheader("Node-by-Node Performance")
                
                # Display metrics for each workflow node
                for node_name, metrics in result['latency_metrics'].items():
                    if isinstance(metrics, dict) and 'total_duration_seconds' in metrics:
                        # Create a metric card for each node
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                f"🕐 {node_name.replace('_', ' ').title()}",
                                f"{metrics.get('total_duration_seconds', 0):.3f}s",
                                help=f"Total execution time for {node_name}"
                            )
                        
                        with col2:
                            llm_duration = metrics.get('llm_duration_seconds', 0)
                            if llm_duration > 0:
                                st.metric(
                                    "🤖 LLM Time",
                                    f"{llm_duration:.3f}s",
                                    help="Time spent in LLM processing"
                                )
                            else:
                                st.metric("🤖 LLM Time", "N/A")
                        
                        with col3:
                            overhead = metrics.get('overhead_duration_seconds', 0)
                            if overhead > 0:
                                st.metric(
                                    "⚙️ Overhead",
                                    f"{overhead:.3f}s",
                                    help="Time spent in data processing and other operations"
                                )
                            else:
                                st.metric("⚙️ Overhead", "N/A")
                        
                        with col4:
                            status = metrics.get('status', 'unknown')
                            status_color = "🟢" if status == "success" else "🔴" if status == "error" else "🟡"
                            st.metric(
                                "📊 Status",
                                f"{status_color} {status.title()}",
                                help=f"Execution status: {status}"
                            )
                        
                        # Add detailed timing breakdown
                        with st.expander(f"📋 Detailed Metrics for {node_name.replace('_', ' ').title()}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Timing Details:**")
                                if 'start_time' in metrics:
                                    st.markdown(f"• **Start:** {metrics['start_time']}")
                                if 'end_time' in metrics:
                                    st.markdown(f"• **End:** {metrics['end_time']}")
                                if 'total_duration_seconds' in metrics:
                                    st.markdown(f"• **Total Duration:** {metrics['total_duration_seconds']:.3f}s")
                                if 'llm_duration_seconds' in metrics:
                                    st.markdown(f"• **LLM Duration:** {metrics['llm_duration_seconds']:.3f}s")
                                if 'overhead_duration_seconds' in metrics:
                                    st.markdown(f"• **Overhead:** {metrics['overhead_duration_seconds']:.3f}s")
                            
                            with col2:
                                st.markdown("**Additional Info:**")
                                if 'response_length' in metrics:
                                    st.markdown(f"• **Response Length:** {metrics['response_length']} chars")
                                if 'error_message' in metrics:
                                    st.markdown(f"• **Error:** {metrics['error_message']}")
                            
                            # Show progress bar for timing breakdown
                            total_time = metrics.get('total_duration_seconds', 0)
                            if total_time > 0:
                                llm_pct = (metrics.get('llm_duration_seconds', 0) / total_time) * 100
                                overhead_pct = (metrics.get('overhead_duration_seconds', 0) / total_time) * 100
                                
                                st.markdown("**Time Distribution:**")
                                st.progress(llm_pct / 100, text=f"LLM Processing: {llm_pct:.1f}%")
                                st.progress(overhead_pct / 100, text=f"Overhead: {overhead_pct:.1f}%")
                        
                        st.markdown("---")
            
            with tab2:
                st.subheader("Token Usage by Node")
                
                # Calculate total tokens across all nodes
                total_input_tokens = 0
                total_output_tokens = 0
                total_tokens = 0
                
                for node_name, metrics in result['latency_metrics'].items():
                    if isinstance(metrics, dict):
                        input_tokens = metrics.get('input_tokens', 0)
                        output_tokens = metrics.get('output_tokens', 0)
                        node_tokens = metrics.get('total_tokens', 0)
                        
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        total_tokens += node_tokens
                        
                        # Display token metrics for each node
                        if input_tokens > 0 or output_tokens > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(f"**{node_name.replace('_', ' ').title()}**")
                            
                            with col2:
                                st.metric("📥 Input", f"{input_tokens:,}")
                            
                            with col3:
                                st.metric("📤 Output", f"{output_tokens:,}")
                            
                            with col4:
                                st.metric("📊 Total", f"{node_tokens:,}")
                            
                            st.markdown("---")
                
                # Display overall token summary
                if total_tokens > 0:
                    st.subheader("📊 Overall Token Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📥 Total Input Tokens", f"{total_input_tokens:,}")
                    
                    with col2:
                        st.metric("📤 Total Output Tokens", f"{total_output_tokens:,}")
                    
                    with col3:
                        st.metric("📊 Total Tokens Used", f"{total_tokens:,}")
                    
                    # Token cost estimation (rough estimate)
                    # GPT-4 pricing: $0.03 per 1K input tokens, $0.06 per 1K output tokens
                    estimated_cost = (total_input_tokens * 0.00003) + (total_output_tokens * 0.00006)
                    st.info(f"💰 Estimated Cost: ${estimated_cost:.4f} (based on GPT-4 pricing)")
            
            with tab3:
                st.subheader("Performance Summary")
                
                # Calculate overall performance metrics
                total_duration = sum(
                    metrics.get('total_duration_seconds', 0) 
                    for metrics in result['latency_metrics'].values() 
                    if isinstance(metrics, dict)
                )
                
                total_llm_time = sum(
                    metrics.get('llm_duration_seconds', 0) 
                    for metrics in result['latency_metrics'].values() 
                    if isinstance(metrics, dict)
                )
                
                total_overhead = sum(
                    metrics.get('overhead_duration_seconds', 0) 
                    for metrics in result['latency_metrics'].values() 
                    if isinstance(metrics, dict)
                )
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("⏱️ Total Workflow Time", f"{total_duration:.3f}s")
                
                with col2:
                    st.metric("🤖 Total LLM Time", f"{total_llm_time:.3f}s")
                
                with col3:
                    st.metric("⚙️ Total Overhead", f"{total_overhead:.3f}s")
                
                # Efficiency metrics
                if total_duration > 0:
                    efficiency = (total_llm_time / total_duration) * 100
                    st.metric("📈 LLM Efficiency", f"{efficiency:.1f}%")
                    
                    # Performance insights
                    st.subheader("💡 Performance Insights")
                    
                    if efficiency > 80:
                        st.success("✅ Excellent performance! Most time is spent in LLM processing, indicating efficient data handling.")
                    elif efficiency > 60:
                        st.info("ℹ️ Good performance with reasonable overhead. Consider optimizing data processing if needed.")
                    else:
                        st.warning("⚠️ High overhead detected. Consider optimizing data retrieval and processing operations.")
                    
                    # Bottleneck analysis
                    st.subheader("🔍 Bottleneck Analysis")
                    node_performance = []
                    for node_name, metrics in result['latency_metrics'].items():
                        if isinstance(metrics, dict) and 'total_duration_seconds' in metrics:
                            node_performance.append((
                                node_name,
                                metrics['total_duration_seconds']
                            ))
                    
                    if node_performance:
                        # Sort by duration to identify bottlenecks
                        node_performance.sort(key=lambda x: x[1], reverse=True)
                        
                        st.markdown("**Slowest Nodes:**")
                        for i, (node_name, duration) in enumerate(node_performance[:3]):
                            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.markdown(f"{icon} **{node_name.replace('_', ' ').title()}**: {duration:.3f}s")
    
    # Add dedicated Workflow Performance section as expandable dropdown
    if result.get('latency_metrics'):
        with st.expander("🚀 Workflow Performance", expanded=False):
            # Calculate comprehensive performance metrics
            total_duration = sum(
                metrics.get('total_duration_seconds', 0) 
                for metrics in result['latency_metrics'].values() 
                if isinstance(metrics, dict)
            )
            
            total_llm_time = sum(
                metrics.get('llm_duration_seconds', 0) 
                for metrics in result['latency_metrics'].values() 
                if isinstance(metrics, dict)
            )
            
            total_overhead = sum(
                metrics.get('overhead_duration_seconds', 0) 
                for metrics in result['latency_metrics'].values() 
                if isinstance(metrics, dict)
            )
            
            # Calculate token metrics
            total_input_tokens = sum(
                metrics.get('input_tokens', 0) 
                for metrics in result['latency_metrics'].values() 
                if isinstance(metrics, dict)
            )
            
            total_output_tokens = sum(
                metrics.get('output_tokens', 0) 
                for metrics in result['latency_metrics'].values() 
                if isinstance(metrics, dict)
            )
            
            total_tokens = total_input_tokens + total_output_tokens
            
            # Performance Overview Cards
            st.subheader("📊 Performance Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "⏱️ Total Duration",
                    f"{total_duration:.3f}s",
                    help="Complete workflow execution time"
                )
            
            with col2:
                st.metric(
                    "🤖 LLM Processing",
                    f"{total_llm_time:.3f}s",
                    help="Time spent in AI model processing"
                )
            
            with col3:
                st.metric(
                    "⚙️ Data Processing",
                    f"{total_overhead:.3f}s",
                    help="Time spent in data retrieval and processing"
                )
            
            with col4:
                if total_duration > 0:
                    efficiency = (total_llm_time / total_duration) * 100
                    st.metric(
                        "📈 Efficiency",
                        f"{efficiency:.1f}%",
                        help="LLM processing efficiency ratio"
                    )
                else:
                    st.metric("📈 Efficiency", "N/A")
            
            # Cost Analysis Section
            st.subheader("💰 Cost Analysis")
            
            # Model pricing information (based on OpenAI pricing)
            model_pricing = {
                "gpt-4o": {"input": 0.000005, "output": 0.000015},  # $5.00 / 1M input, $15.00 / 1M output
                "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},  # $0.15 / 1M input, $0.60 / 1M output
                "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},  # $0.50 / 1M input, $1.50 / 1M output
            }
            
            # Get the model from the workflow (you might need to pass this from the main function)
            # For now, we'll show costs for all models
            cost_col1, cost_col2, cost_col3 = st.columns(3)
            
            with cost_col1:
                st.markdown("**GPT-4o (Most Capable)**")
                gpt4o_cost = (total_input_tokens * model_pricing["gpt-4o"]["input"]) + (total_output_tokens * model_pricing["gpt-4o"]["output"])
                st.metric("💰 Cost", f"${gpt4o_cost:.6f}")
                st.markdown(f"📥 Input: ${total_input_tokens * model_pricing['gpt-4o']['input']:.6f}")
                st.markdown(f"📤 Output: ${total_output_tokens * model_pricing['gpt-4o']['output']:.6f}")
            
            with cost_col2:
                st.markdown("**GPT-4o-mini (Balanced)**")
                gpt4o_mini_cost = (total_input_tokens * model_pricing["gpt-4o-mini"]["input"]) + (total_output_tokens * model_pricing["gpt-4o-mini"]["output"])
                st.metric("💰 Cost", f"${gpt4o_mini_cost:.6f}")
                st.markdown(f"📥 Input: ${total_input_tokens * model_pricing['gpt-4o-mini']['input']:.6f}")
                st.markdown(f"📤 Output: ${total_output_tokens * model_pricing['gpt-4o-mini']['output']:.6f}")
            
            with cost_col3:
                st.markdown("**GPT-3.5-turbo (Fastest)**")
                gpt35_cost = (total_input_tokens * model_pricing["gpt-3.5-turbo"]["input"]) + (total_output_tokens * model_pricing["gpt-3.5-turbo"]["output"])
                st.metric("💰 Cost", f"${gpt35_cost:.6f}")
                st.markdown(f"📥 Input: ${total_input_tokens * model_pricing['gpt-3.5-turbo']['input']:.6f}")
                st.markdown(f"📤 Output: ${total_output_tokens * model_pricing['gpt-3.5-turbo']['output']:.6f}")
            
            # Cost optimization insights
            st.subheader("💡 Cost Optimization Insights")
            
            if total_tokens > 0:
                # Find the most cost-effective model
                costs = {
                    "GPT-4o": gpt4o_cost,
                    "GPT-4o-mini": gpt4o_mini_cost,
                    "GPT-3.5-turbo": gpt35_cost
                }
                
                most_affordable = min(costs, key=costs.get)
                most_expensive = max(costs, key=costs.get)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"✅ **Most Affordable:** {most_affordable} (${costs[most_affordable]:.6f})")
                    st.info(f"💡 **Cost Savings:** Using {most_affordable} instead of {most_expensive} saves ${costs[most_expensive] - costs[most_affordable]:.6f}")
                
                with col2:
                    st.warning(f"⚠️ **Most Expensive:** {most_expensive} (${costs[most_expensive]:.6f})")
                    st.info(f"📊 **Token Efficiency:** {total_tokens:,} tokens processed")
            
            # Performance benchmarks
            st.subheader("🏆 Performance Benchmarks")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if total_duration > 0:
                    tokens_per_second = total_tokens / total_duration
                    st.metric("🚀 Processing Speed", f"{tokens_per_second:.1f} tokens/s")
                else:
                    st.metric("🚀 Processing Speed", "N/A")
            
            with col2:
                if total_llm_time > 0:
                    llm_tokens_per_second = total_tokens / total_llm_time
                    st.metric("🤖 LLM Speed", f"{llm_tokens_per_second:.1f} tokens/s")
                else:
                    st.metric("🤖 LLM Speed", "N/A")
            
            with col3:
                if total_tokens > 0:
                    cost_per_token = gpt4o_mini_cost / total_tokens  # Using balanced model as reference
                    st.metric("💰 Cost per Token", f"${cost_per_token:.8f}")
                else:
                    st.metric("💰 Cost per Token", "N/A")
            
            # Detailed breakdown
            with st.expander("📋 Detailed Performance Breakdown"):
                st.markdown("**Node-by-Node Analysis:**")
                
                for node_name, metrics in result['latency_metrics'].items():
                    if isinstance(metrics, dict) and 'total_duration_seconds' in metrics:
                        node_duration = metrics.get('total_duration_seconds', 0)
                        node_llm_time = metrics.get('llm_duration_seconds', 0)
                        node_input_tokens = metrics.get('input_tokens', 0)
                        node_output_tokens = metrics.get('output_tokens', 0)
                        
                        if node_duration > 0:
                            st.markdown(f"**{node_name.replace('_', ' ').title()}:**")
                            st.markdown(f"  • Duration: {node_duration:.3f}s")
                            st.markdown(f"  • LLM Time: {node_llm_time:.3f}s ({(node_llm_time/node_duration)*100:.1f}%)")
                            st.markdown(f"  • Tokens: {node_input_tokens + node_output_tokens:,} (Input: {node_input_tokens:,}, Output: {node_output_tokens:,})")
                            st.markdown("---")

    # Display performance metrics
    if result.get('performance_metrics'):
        st.header("⚡ Performance Metrics")
        
        metrics = result['performance_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_duration' in metrics:
                st.metric("Total Duration", f"{metrics['total_duration']:.2f}s")
        
        with col2:
            if 'llm_calls' in metrics:
                st.metric("LLM Calls", metrics['llm_calls'])
        
        with col3:
            if 'tokens_used' in metrics:
                st.metric("Tokens Used", metrics['tokens_used'])

def create_sample_data():
    """Create sample data for demonstration"""
    st.header("📊 Sample Data Overview")
    
    try:
        data_loader = DataLoader()
        data_loader.load_all_data()
        
        # Display data summaries
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(data_loader, 'aws_data') and data_loader.aws_data is not None:
                st.markdown("**AWS Cost Data**")
                st.dataframe(data_loader.aws_data.head(), use_container_width=True)
                st.markdown(f"Shape: {data_loader.aws_data.shape}")
        
        with col2:
            if hasattr(data_loader, 'budget_data') and data_loader.budget_data is not None:
                st.markdown("**Budget Data**")
                st.dataframe(data_loader.budget_data.head(), use_container_width=True)
                st.markdown(f"Shape: {data_loader.budget_data.shape}")
        
        with col3:
            if hasattr(data_loader, 'vendor_data') and data_loader.vendor_data is not None:
                st.markdown("**Vendor Data**")
                st.dataframe(data_loader.vendor_data.head(), use_container_width=True)
                st.markdown(f"Shape: {data_loader.vendor_data.shape}")
    
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

if __name__ == "__main__":
    main()
