# Auto-Spend: LangGraph Workflow for Financial Data Analysis

A sophisticated LangGraph workflow that automatically classifies financial questions and retrieves relevant data from multiple datasets to provide comprehensive spending analysis.

## 🏗️ Architecture

The system consists of three main components:

1. **DataLoader** (`dataloader.py`) - Loads and manages CSV/JSON data files
2. **LangGraph Workflow** (`langgraph_workflow.py`) - Core workflow with classification and retrieval agents
3. **Main Runner** (`run_workflow.py`) - Executes the workflow with example questions and interactive mode

## 🚀 Features

- **Automatic Question Classification**: Uses OpenAI to classify questions into three categories:
  - `aws_costs` - AWS cloud infrastructure costs
  - `budget` - Project and team budget tracking
  - `vendor_spend` - Vendor and supplier expenses

- **Intelligent Data Retrieval**: Retrieves relevant data based on classification
- **Professional Response Generation**: Creates executive-ready answers with structured format:
  - **Executive Summary** - Key findings overview
  - **Detailed Analysis** - Specific metrics and data insights
  - **Key Insights** - Business implications and takeaways
  - **Recommendations** - Actionable next steps and strategies
  - **Data Sources** - Reference to datasets used
- **Enhanced Data Formatting**: Professional presentation with currency symbols and organized structure
- **Multi-format Support**: Handles CSV and JSON data files
- **Interactive Mode**: Ask your own questions in real-time

## 🔍 Anomaly Detection & Analysis

The system provides comprehensive anomaly detection capabilities:

### **Statistical Analysis**
- **Mean, Standard Deviation, Min/Max** calculations for trend analysis
- **Variance Analysis** with percentage calculations
- **Range Analysis** to identify unusual deviations
- **Pattern Recognition** across time series data

### **Anomaly Identification**
- **Unusual Spikes/Drops** in spending patterns
- **Outlier Detection** using statistical methods
- **Trend Deviations** from normal patterns
- **Budget Variance Analysis** with risk assessment

### **Business Impact Assessment**
- **Risk Level Classification** for identified anomalies
- **Business Impact Analysis** of unusual patterns
- **Confidence Levels** in anomaly detection
- **Data Quality Assessment** for reliable analysis

### **Actionable Insights**
- **Specific Recommendations** for addressing anomalies
- **Risk Mitigation Strategies** for unusual patterns
- **Optimization Opportunities** based on findings
- **Preventive Measures** for future anomalies

## 📊 Supported Data Types

The system automatically loads and analyzes:

- **AWS Costs**: Daily cloud infrastructure costs by service
- **Budget Tracking**: Project budgets, actual spend, and variance analysis
- **Vendor Data**: Vendor contracts, spending, and risk assessment

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd auto-spend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key
   ```

## 🔑 Configuration

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_actual_api_key_here
```

Get your API key from: [OpenAI Platform](https://platform.openai.com/api-keys)

## 🎯 Usage

### Quick Start

Run the workflow and start asking questions immediately:

```bash
python run_workflow.py
```

### Interactive Mode

The system starts directly in interactive mode where you can ask your own questions:

```
❓ Your question: What are our total AWS costs this year?
❓ Your question: Which projects are over budget?
❓ Your question: How much do we spend on vendors?
```



## 🔄 Workflow Process

1. **Question Input**: User asks a financial question
2. **Classification**: OpenAI classifies the question into one of three categories
3. **Data Retrieval**: Relevant data is extracted from the appropriate dataset
4. **Answer Generation**: Professional, structured answer is generated using the retrieved data
5. **Output**: Executive-ready response with clear sections and actionable insights

## 📋 Response Format

The system generates professional, structured responses with the following sections:

### **EXECUTIVE SUMMARY**
- Brief overview of key findings and main answer to the question

### **DETAILED ANALYSIS**
- Specific data metrics and numbers from the datasets
- **ANOMALY DETECTION**: Identification and explanation of unusual spikes, drops, or patterns
- Trends, patterns, and data quality issues
- Comparative analysis where available
- Statistical analysis with confidence levels

### **KEY INSIGHTS**
- 2-3 bullet points of the most important takeaways
- Business implications and impact assessment
- **ANOMALY IMPACT**: Business impact of any detected anomalies

### **RECOMMENDATIONS**
- Actionable next steps and strategies
- Risk mitigation approaches
- Optimization opportunities
- **ANOMALY RESPONSE**: Specific actions to address detected anomalies

### **DATA SOURCES & QUALITY**
- Reference to the specific datasets and data used for analysis
- Data quality assessment and confidence levels
- Completeness and reliability indicators

## 📁 File Structure

```
auto-spend/
├── docs/                          # Data files
│   ├── daily-aws-costs.csv       # AWS cost data
│   ├── sample-budget-tracking.csv # Budget tracking data
│   ├── sample-vendor-data.csv    # Vendor spending data
│   └── sample-vendor-data.json   # Vendor data (JSON format)
├── dataloader.py                 # Data loading and management
├── langgraph_workflow.py         # Main LangGraph workflow
├── run_workflow.py               # Execution script
├── requirements.txt               # Python dependencies
├── env_example.txt               # Environment variables template
└── README.md                     # This file
```

## 🧪 Testing

The system automatically tests with example questions covering all three categories. You can also test individual components:

```python
from dataloader import DataLoader
from langgraph_workflow import SpendAnalyzerWorkflow

# Test data loading
loader = DataLoader()
data = loader.load_all_data()
print(f"Loaded {len(data)} datasets")

# Test workflow
workflow = SpendAnalyzerWorkflow()
result = workflow.run("What are our AWS costs?")
print(result)
```

### Testing Enhanced Features

Test the enhanced response format and prompts:

```bash
# Test component functionality
python3 test_components.py

# Test enhanced response format
python3 test_response_format.py

# Test anomaly detection capabilities
python3 test_anomaly_detection.py

# Run demo without API key
python3 demo_workflow.py
```

### Response Format Testing

The enhanced response format includes:
- **Structured sections** with clear headings
- **Professional data formatting** with currency symbols
- **Executive-ready language** and business context
- **Actionable insights** and recommendations

### Anomaly Detection Testing

The enhanced anomaly detection includes:
- **Complete data structure access** for comprehensive analysis
- **Statistical metrics** (mean, std dev, min/max, ranges)
- **Pattern recognition** and trend analysis
- **Business impact assessment** of anomalies
- **Specific recommendations** for anomaly response

## 🔧 Customization

### Adding New Data Sources

1. Place new CSV/JSON files in the `docs/` folder
2. The system automatically detects and loads them
3. Update the retrieval methods in `langgraph_workflow.py` if needed

### Modifying Classification

Edit the classification prompt in `_classify_question()` method to add new categories or modify existing ones. The enhanced prompt includes detailed descriptions for better accuracy.

### Custom Retrieval Logic

Modify the `_retrieve_*_data()` methods to implement custom data extraction logic.

### Response Format Customization

The response format can be customized by modifying:
- **System Message**: Change the overall tone and expertise level in `__init__()`
- **Answer Prompt**: Modify the structure and sections in `_generate_answer()`
- **Data Formatting**: Adjust how data is presented in `_format_data_for_prompt()`

### Enhanced Prompts

The system uses enhanced prompts for:
- **Classification**: More detailed category descriptions for better accuracy
- **Answer Generation**: Structured format with executive summary, analysis, insights, and recommendations
- **Data Presentation**: Professional formatting with currency symbols and organized structure

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure `.env` file exists with correct API key
   - Verify API key is valid and has sufficient credits

2. **Data Loading Errors**:
   - Check that `docs/` folder contains the required data files
   - Verify file formats (CSV/JSON) are correct

3. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Debug Mode

Add debug prints in the workflow methods to trace execution:

```python
print(f"Processing classification: {state['classification']}")
print(f"Retrieved data: {relevant_data}")
```

## 📈 Performance

- **Classification**: ~1-2 seconds per question
- **Data Retrieval**: ~0.1-0.5 seconds
- **Answer Generation**: ~2-5 seconds
- **Total Response Time**: ~3-8 seconds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/) GPT models
- Data analysis with [Pandas](https://pandas.pydata.org/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the example questions and responses
3. Open an issue on the repository
4. Check the LangGraph documentation for advanced usage
