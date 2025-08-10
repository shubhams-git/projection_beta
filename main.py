from google import genai
from google.genai import types
import pathlib
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

client = genai.Client()

class GrowthRateAssumptions(BaseModel):
    """Growth rate assumptions with fixed fields"""
    revenue_cagr: float = Field(description="Revenue compound annual growth rate")
    expense_inflation: float = Field(description="Expense inflation rate")
    profit_margin_target: float = Field(description="Target profit margin")

class KeyFinancialRatios(BaseModel):
    """Key financial ratios with fixed fields"""
    gross_margin: float = Field(description="Gross profit margin")
    net_margin: float = Field(description="Net profit margin")
    current_ratio: float = Field(description="Current assets / Current liabilities")
    debt_to_equity: float = Field(description="Total debt / Total equity")

class MonthlyProjection(BaseModel):
    """Monthly projection data structure"""
    month: str = Field(description="Format: 2026-01, 2026-02, etc.")
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class QuarterlyProjection(BaseModel):
    """Quarterly projection data structure"""
    quarter: str = Field(description="Format: 2026-Q1, 2026-Q2, etc.")
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class AnnualProjection(BaseModel):
    """Annual projection data structure"""
    year: int
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class ProjectionsData(BaseModel):
    """Comprehensive projections data structure"""
    one_year_monthly: List[MonthlyProjection] = Field(description="12 months of monthly data")
    three_years_monthly: List[MonthlyProjection] = Field(description="36 months of monthly data")
    five_years_quarterly: List[QuarterlyProjection] = Field(description="20 quarters of quarterly data")
    ten_years_annual: List[AnnualProjection] = Field(description="10 years of annual data")
    fifteen_years_annual: List[AnnualProjection] = Field(description="15 years of annual data")

class MethodologyDetails(BaseModel):
    """Details about the projection methodology used"""
    forecasting_methods_used: List[str]
    seasonal_adjustments_applied: bool
    trend_analysis_period: str
    growth_rate_assumptions: GrowthRateAssumptions

class ProjectionSchema(BaseModel):
    """Enhanced schema for comprehensive financial projections"""
    executive_summary: str
    business_name: str
    completion_score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    data_quality_score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    projection_confidence_score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    projection_drivers_found: List[str]
    assumptions_made: List[str]
    anomalies_found: List[str]
    methodology: MethodologyDetails
    projections_data: ProjectionsData
    
    # Additional useful fields for business analysis
    key_financial_ratios: KeyFinancialRatios
    risk_factors: List[str]
    recommendations: List[str]

# Retrieve and encode the CSV files
filepath1 = pathlib.Path('Profit and Loss - MJV Plumbing Services.csv')
filepath2 = pathlib.Path('Balance Sheet - MJV Plumbing Services.csv')

prompt = """
Use your full potential of Deep Think, Deep Research and other relevant capabilities to analyse both the profit and loss as well as balance sheet data linked. 

PROJECTION REQUIREMENTS:
Provide detailed financial projections for Revenue, Net Profit, Gross Profit, and Expenses across these timeframes:
- 1 year: Monthly values (12 data points)
- 3 years: Monthly values (36 data points) 
- 5 years: Quarterly values (20 data points)
- 10 years: Annual values (10 data points)
- 15 years: Annual values (15 data points)

ANALYSIS METHODOLOGY:
1. Identify historical trends and seasonality patterns
2. Calculate growth rates and financial ratios
3. Apply appropriate forecasting techniques (trend analysis, seasonal decomposition, regression)
4. Consider industry benchmarks and economic factors
5. Account for business lifecycle stage and market conditions
6. Validate projections against realistic business constraints

CRITICAL DATA CONTEXT:
Different businesses use varying chart of accounts, but certain high-level categories are universally present. There could be multiple sub-fields that maybe the sub-part of the fields given below and when these subfields get added they give the sum of the following fields. Your task is to extract ALL available data and map it to these GUARANTEED STANDARD FIELDS that exist across all businesses:

GUARANTEED P&L STANDARD FIELDS (1-10 fields given below - only include if document contains P&L data):
1. Revenue (Sales, Turnover, Income, Total Revenue)
2. Cost of Sales (COGS, Cost of Goods Sold, Direct Costs)
3. Gross Profit (Gross Margin, Gross Income)
4. Operating Expenses (Total Expenses, Total OpEx, Overhead, Admin Expenses)
5. Operating Profit (EBIT, Operating Income, EBITDA before D&A)
6. Interest Expenses (Finance Costs, Interest Paid, Borrowing Costs)
7. Earnings Before Tax (EBT, Profit Before Tax, Pre-tax Income)
8. Tax Expenses (Income Tax, Tax Provision, Corporate Tax)
9. Earnings After Tax (EAT, Profit After Tax, After-tax Income)
10. Net Income (Net Profit, Bottom Line, Final Profit)

GUARANTEED BALANCE SHEET STANDARD FIELDS (11-25 fields given below - only include if document contains BS data):
ASSETS:
11. Cash & Cash Equivalents (Cash, Bank, Liquid Assets, Short-term Investments)
12. Accounts Receivable (Trade Debtors, AR, Customer Receivables)
13. Inventory (Stock, Work in Progress, Finished Goods)
14. Total Current Assets (Current Assets, Short-term Assets)
15. Fixed Assets Net (PPE Net, Property Plant Equipment, Non-current Assets)
16. Total Assets

LIABILITIES:
17. Accounts Payable (Trade Creditors, AP, Supplier Payables)
18. Short Term Debt (Current Portion Debt, Bank Overdraft, Current Borrowings)
19. Total Current Liabilities (Current Liabilities, Short-term Liabilities)
20. Long Term Debt (Long-term Borrowings, Non-current Debt)
21. Total Liabilities

EQUITY:
22. Share Capital (Paid-in Capital, Issued Capital, Owner's Capital)
23. Retained Earnings (Accumulated Profits, Reserves, Undistributed Profits)
24. Total Equity (Shareholders' Equity, Owner's Equity, Net Worth)
25. Total Liabilities and Equity

RESPONSE REQUIREMENTS:
- Ensure all projections are realistic and defensible
- Include confidence intervals where appropriate
- Explain methodology and key assumptions clearly
- Highlight any data limitations or uncertainties
- Provide actionable insights and recommendations
- Structure response according to the enhanced ProjectionSchema
"""

try:
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(
                data=filepath1.read_bytes(),
                mime_type='text/csv',
            ),
            types.Part.from_bytes(
                data=filepath2.read_bytes(),
                mime_type='text/csv',
            ),
            prompt
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ProjectionSchema,
        },
    )

    print(response.text)

    if response.usage_metadata:
        print(f"Input Token({response.usage_metadata.prompt_token_count}) -> Output Token ({response.usage_metadata.candidates_token_count}) | Total: {response.usage_metadata.total_token_count}")
    else:
        print("Usage metadata not available in the response.")
        print(f"Full response object: {response}")

    # Optional: Parse and validate the JSON response
    try:
        import json
        parsed_response = json.loads(response.text)
        validated_projection = ProjectionSchema(**parsed_response)
        print("\n✅ Schema validation successful!")
        print(f"Business: {validated_projection.business_name}")
        print(f"Confidence Score: {validated_projection.projection_confidence_score:.2f}")
        print(f"Data Quality Score: {validated_projection.data_quality_score:.2f}")
    except Exception as e:
        print(f"\n❌ Schema validation failed: {e}")

except Exception as e:
    print(f"Error during API call: {e}")
    print("This might be due to file not found, API key issues, or network problems.")