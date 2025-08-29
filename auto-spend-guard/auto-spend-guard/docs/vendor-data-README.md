# Vendor Data Documentation

This folder contains comprehensive vendor data for the Auto-Spend-Guard FinOps system, providing detailed information about all vendors, their usage, budgets, and ownership.

## Files

### 1. `sample-vendor-data.json`
Comprehensive vendor information in JSON format with detailed nested structures.

### 2. `sample-vendor-data.csv`
Flattened vendor data in CSV format for easy analysis and import into other tools.

## Data Structure

Each vendor record contains the following key information:

### Basic Information
- **Vendor ID**: Unique identifier for each vendor
- **Vendor Name**: Company name
- **Vendor Type**: Category (e.g., Cloud Infrastructure, Development Tools)
- **Status**: Active/Inactive
- **Risk Level**: Low/Medium/High

### Financial Information
- **Annual Budget Approved**: Total budget allocated for the year
- **Monthly Budget**: Monthly budget allocation
- **Current Annual Spend**: Actual spending to date
- **Budget Variance**: Difference between budget and actual spend
- **Budget Variance Percentage**: Variance as a percentage

### Contract Details
- **Contract Number**: Internal contract identifier
- **Start/End Date**: Contract duration
- **Renewal Date**: When contract needs renewal
- **Payment Terms**: Billing frequency and terms
- **Contract Type**: Enterprise Agreement, Professional, etc.

### Usage Information
- **Usage Purpose**: What the vendor is used for
- **Business Justification**: Why this vendor was chosen
- **Primary Users**: Teams and user counts using the service
- **Usage Reason**: Specific reasons each team uses the service

### Ownership & Approval
- **Owner**: Person responsible for vendor relationship
- **Owner Title**: Job title of the owner
- **Owner Department**: Department the owner belongs to
- **Owner Contact**: Email and phone of the owner
- **Approver**: Person who approved the vendor contract
- **Approver Title**: Job title of the approver

### Contact Information
- **Primary Contact**: Vendor representative details
- **Contact Email**: Vendor contact email
- **Contact Phone**: Vendor contact phone number

## Sample Questions This Data Can Answer

### Financial Analysis
- What is our total vendor spend this year?
- Which vendors are over/under budget?
- What is the budget variance for each vendor?
- How much do we spend on each vendor category?

### Vendor Management
- Who owns each vendor relationship?
- When do contracts need renewal?
- Which vendors are high/medium/low risk?
- What is the business justification for each vendor?

### Usage Analysis
- Which teams use each vendor?
- How many users does each vendor have?
- What is the primary purpose of each vendor?
- Which vendors serve multiple teams?

### Contract Management
- What are the payment terms for each vendor?
- When do contracts expire?
- What type of contracts do we have?
- Who approved each vendor contract?

## Data Relationships

The vendor data connects to other FinOps data:
- **Cloud Costs**: AWS, Azure, GCP costs from `sample-cloud-costs.json`
- **SaaS Costs**: Slack, GitHub, Salesforce costs from `sample-saas-costs.csv`
- **Budget Tracking**: Team budgets from `sample-budget-tracking.csv`
- **Cost Optimization**: Vendor-specific optimization opportunities

## Usage Examples

### 1. Vendor Spend Analysis
```json
// Find vendors over budget
vendors.filter(v => v.budget_variance < 0)
```

### 2. Team Ownership Mapping
```json
// Map vendors to owners
vendors.map(v => ({
  vendor: v.vendor_name,
  owner: v.owner.name,
  department: v.owner.department
}))
```

### 3. Contract Renewal Planning
```json
// Find contracts expiring soon
vendors.filter(v => new Date(v.contract_details.end_date) < new Date('2024-06-30'))
```

### 4. Risk Assessment
```json
// High-risk vendors
vendors.filter(v => v.risk_level === 'High')
```

## Data Quality Notes

- All monetary values are in USD
- Dates use ISO 8601 format (YYYY-MM-DD)
- Contact information is fictional for privacy
- Budgets and costs are illustrative examples
- Vendor relationships are realistic but fictional

## Integration Points

This vendor data can be integrated with:
- **Cost Monitoring Systems**: Track actual vs. budgeted spend
- **Contract Management**: Alert on renewal dates
- **Risk Management**: Flag high-risk vendors
- **Team Budgeting**: Allocate costs to teams
- **Vendor Performance**: Track SLA compliance and issues
