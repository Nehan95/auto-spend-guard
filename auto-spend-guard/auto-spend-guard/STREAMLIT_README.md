# 🚀 Auto-Spend Guard Streamlit App

A beautiful, interactive web interface for the Auto-Spend Guard workflow that makes financial analysis accessible to everyone.

## ✨ Features

- **🎯 Interactive Analysis**: Ask questions in natural language about your financial data
- **📊 Real-time Visualizations**: Beautiful charts and graphs powered by Plotly
- **🔧 Configurable AI Models**: Choose between different OpenAI models and adjust creativity
- **📈 Quick Insights**: Pre-built analysis templates for common financial questions
- **💾 Analysis History**: Track and review your previous analyses
- **🎨 Modern UI**: Clean, professional interface with responsive design

## 🚀 Quick Start

### Option 1: Use the Launcher Script (Recommended)

```bash
# Make the script executable
chmod +x run_streamlit.py

# Run the app
python3 run_streamlit.py
```

### Option 2: Manual Streamlit Command

```bash
# Install dependencies first
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## 📱 How to Use

### 1. **Configuration Sidebar**
- **AI Model Selection**: Choose between GPT-4o-mini, GPT-4o, or GPT-3.5-turbo
- **Temperature Control**: Adjust AI creativity (0.0 = focused, 1.0 = creative)
- **Analysis Type**: Select the type of analysis you want to perform
- **Advanced Options**: Enable/disable visualizations and debug mode

### 2. **Query Input**
- **Natural Language**: Ask questions like "Find unusual spending patterns" or "Analyze our AWS costs"
- **Quick Queries**: Use pre-built buttons for common analyses
- **Custom Questions**: Write your own specific financial analysis questions

### 3. **Results & Visualizations**
- **Analysis Summary**: AI-generated insights and recommendations
- **Interactive Charts**: Hover over data points for detailed information
- **Performance Metrics**: Track analysis execution time and resource usage
- **Debug Information**: Detailed workflow execution details (when enabled)

## 🎯 Example Queries

### **Anomaly Detection**
```
"Identify any unusual spending patterns or anomalies in our financial data"
```

### **Cost Optimization**
```
"Analyze our AWS spending patterns and provide cost optimization recommendations"
```

### **Budget Analysis**
```
"Review our budget tracking data and identify areas for improvement"
```

### **Vendor Analysis**
```
"Analyze our vendor spending patterns and identify potential cost savings"
```

### **Trend Analysis**
```
"What are the spending trends over the last quarter and what do they indicate?"
```

## 🔧 Configuration Options

### **AI Model Settings**
- **GPT-4o-mini**: Fast, cost-effective analysis (default)
- **GPT-4o**: More detailed and accurate analysis
- **GPT-3.5-turbo**: Balanced performance and cost

### **Temperature Control**
- **0.0-0.3**: Highly focused, consistent responses
- **0.4-0.7**: Balanced creativity and focus (recommended)
- **0.8-1.0**: More creative, varied responses

### **Advanced Features**
- **Workflow Visualization**: Enable/disable interactive charts
- **Debug Mode**: Show detailed workflow execution information
- **Max Tokens**: Control response length and detail level

## 📊 Data Sources

The app automatically loads and analyzes:

- **AWS Cost Data**: Daily spending patterns and service breakdowns
- **Budget Tracking**: Planned vs. actual spending by category
- **Vendor Information**: Supplier costs and performance metrics

## 🎨 Customization

### **Styling**
The app uses custom CSS for a professional appearance:
- Gradient headers
- Color-coded metric cards
- Responsive layout for different screen sizes

### **Charts**
Interactive visualizations powered by Plotly:
- Line charts for time series data
- Pie charts for service breakdowns
- Bar charts for budget comparisons
- Hover tooltips for detailed information

## 🚨 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Kill existing process on port 8501
   lsof -ti:8501 | xargs kill -9
   ```

3. **Browser Not Opening**
   - Manually navigate to `http://localhost:8501`
   - Check firewall settings

4. **Data Loading Issues**
   - Ensure data files are in the `docs/` directory
   - Check file permissions and formats

### **Performance Tips**

- Use GPT-4o-mini for quick analyses
- Enable debug mode only when needed
- Close unused browser tabs to free memory

## 🔒 Security Notes

- The app runs locally on your machine
- No data is sent to external servers (except OpenAI API calls)
- API keys are stored locally in environment variables
- Use `.env` file for sensitive configuration

## 📚 API Reference

### **Main Functions**

- `main()`: Main Streamlit application entry point
- `run_analysis()`: Execute the analysis workflow
- `display_results()`: Show analysis results and visualizations
- `create_sample_data()`: Display data overview

### **Session State Variables**

- `analysis_history`: List of previous analyses
- `user_query`: Current user input query

## 🤝 Contributing

To enhance the Streamlit app:

1. **UI Improvements**: Modify the CSS styling and layout
2. **New Visualizations**: Add more chart types and data views
3. **Additional Features**: Implement new analysis capabilities
4. **Performance**: Optimize data loading and processing

## 📄 License

This Streamlit app is part of the Auto-Spend Guard project and follows the same licensing terms.

---

**🎉 Enjoy your interactive financial analysis experience!**
