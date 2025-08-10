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
    reasons_for_low_quality_or_confidence_score: List[str]
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
Use your full potential of Deep Think, reason and other relevant capabilities to analyse both the profit and loss as well as balance sheet data linked. 

PROJECTION REQUIREMENTS:
Provide detailed financial projections for Revenue, Net Profit, Gross Profit, and Expenses across these timeframes (commencing January of Next Year):
- 1 year: Monthly values (12 data points)
- 3 years: Monthly values (36 data points) 
- 5 years: Quarterly values (20 data points)
- 10 years: Annual values (10 data points)
- 15 years: Annual values (15 data points)

RESPONSE REQUIREMENTS:
- Ensure all projections are realistic and defensible
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
            "temperature":0.7
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
        if response.text:
            parsed_response = json.loads(response.text)
            validated_projection = ProjectionSchema(**parsed_response)
            print("\n✅ Schema validation successful!")
            print(f"Business: {validated_projection.business_name}")
            print(f"Confidence Score: {validated_projection.projection_confidence_score:.2f}")
            print(f"Data Quality Score: {validated_projection.data_quality_score:.2f}")
        else:
            print("\n❌ Response text is empty, skipping schema validation.")
    except Exception as e:
        print(f"\n❌ Schema validation failed: {e}")

except Exception as e:
    print(f"Error during API call: {e}")
    print("This might be due to file not found, API key issues, or network problems.")