from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import pathlib
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Projection API", version="1.0.0")

client = genai.Client()

class FinancialGrowthAssumptions(BaseModel):
    """Detailed growth rate assumptions for comprehensive financial modeling"""
    revenue_cagr: float = Field(
        description="Revenue compound annual growth rate as a decimal (e.g., 0.15 for 15% CAGR). Should reflect industry benchmarks, historical performance, and market conditions."
    )
    expense_inflation: float = Field(
        description="Annual expense inflation rate as a decimal (e.g., 0.03 for 3%). Consider cost-push factors, wage inflation, and operational cost trends."
    )
    profit_margin_target: float = Field(
        description="Target net profit margin as a decimal (e.g., 0.12 for 12%). Should be achievable based on industry standards and operational efficiency improvements."
    )

class ComprehensiveFinancialRatios(BaseModel):
    """Critical financial health indicators for business performance assessment"""
    gross_margin: float = Field(
        description="Gross profit margin as a decimal (Revenue - COGS)/Revenue. Indicates pricing power and cost control effectiveness."
    )
    net_margin: float = Field(
        description="Net profit margin as a decimal (Net Income/Revenue). Reflects overall operational efficiency and profitability after all expenses."
    )
    current_ratio: float = Field(
        description="Current assets divided by current liabilities. Values above 1.0 indicate good short-term liquidity. Industry benchmark typically 1.2-2.0."
    )
    debt_to_equity: float = Field(
        description="Total debt divided by total shareholders' equity. Lower ratios indicate less financial risk. Optimal range varies by industry."
    )

class DetailedMonthlyProjection(BaseModel):
    """Granular monthly financial projections with seasonal considerations"""
    month: str = Field(
        description="Month in YYYY-MM format (e.g., '2026-01', '2026-02'). Ensure chronological ordering starting from next January."
    )
    revenue: float = Field(
        description="Total monthly revenue in base currency. Consider seasonal patterns, market cycles, and growth trajectories."
    )
    net_profit: float = Field(
        description="Monthly net profit after all expenses, taxes, and deductions. Should align with projected profit margins."
    )
    gross_profit: float = Field(
        description="Monthly gross profit (Revenue - Cost of Goods Sold). Must be mathematically consistent with revenue and expense projections."
    )
    expenses: float = Field(
        description="Total monthly operating expenses including COGS, SG&A, interest, and taxes. Factor in inflation and scale effects."
    )

class DetailedQuarterlyProjection(BaseModel):
    """Comprehensive quarterly financial projections for medium-term planning"""
    quarter: str = Field(
        description="Quarter in YYYY-QN format (e.g., '2026-Q1', '2026-Q2'). Aggregate monthly data appropriately for quarterly reporting."
    )
    revenue: float = Field(
        description="Total quarterly revenue aggregated from monthly projections. Ensure consistency with seasonal business patterns."
    )
    net_profit: float = Field(
        description="Quarterly net profit reflecting operational performance and one-time adjustments if applicable."
    )
    gross_profit: float = Field(
        description="Quarterly gross profit demonstrating core business profitability before operating expenses."
    )
    expenses: float = Field(
        description="Total quarterly expenses including both fixed and variable costs, scaled appropriately for business growth."
    )

class DetailedAnnualProjection(BaseModel):
    """Strategic annual financial projections for long-term business planning"""
    year: int = Field(
        description="Calendar year for the projection (e.g., 2026, 2027). Should follow logical progression from base year."
    )
    revenue: float = Field(
        description="Annual revenue reflecting cumulative growth, market expansion, and business development initiatives."
    )
    net_profit: float = Field(
        description="Annual net profit incorporating full-year operational results, tax implications, and strategic investments."
    )
    gross_profit: float = Field(
        description="Annual gross profit demonstrating core business unit economics and scale efficiencies."
    )
    expenses: float = Field(
        description="Total annual expenses including operational costs, capital expenditures, and growth investments."
    )

class ComprehensiveProjectionsDataset(BaseModel):
    """Complete financial projections dataset covering all required timeframes"""
    one_year_monthly: List[DetailedMonthlyProjection] = Field(
        description="Exactly 12 months of detailed monthly projections starting from January of next year. Critical for cash flow management and short-term planning."
    )
    three_years_monthly: List[DetailedMonthlyProjection] = Field(
        description="Exactly 36 months of monthly projections covering three full years. Essential for medium-term strategic planning and investor presentations."
    )
    five_years_quarterly: List[DetailedQuarterlyProjection] = Field(
        description="Exactly 20 quarters (5 years) of quarterly projections. Standard timeframe for business plan financial modeling and loan applications."
    )
    ten_years_annual: List[DetailedAnnualProjection] = Field(
        description="Exactly 10 years of annual projections for long-term strategic planning. Consider major market shifts and competitive dynamics."
    )
    fifteen_years_annual: List[DetailedAnnualProjection] = Field(
        description="Exactly 15 years of annual projections for comprehensive long-term analysis. Factor in industry lifecycle and technological disruption potential."
    )

class ProjectionMethodologyFramework(BaseModel):
    """Comprehensive methodology documentation for projection transparency and validation"""
    forecasting_methods_used: List[str] = Field(
        description="List of specific forecasting techniques applied (e.g., 'Trend Analysis', 'Regression Modeling', 'Seasonal Decomposition', 'Monte Carlo Simulation')."
    )
    seasonal_adjustments_applied: bool = Field(
        description="Whether seasonal patterns were identified and incorporated into projections. Critical for businesses with cyclical revenue patterns."
    )
    trend_analysis_period: str = Field(
        description="Time period used for historical trend analysis (e.g., '3 years', '5 years'). Longer periods provide more stability but may miss recent shifts."
    )
    growth_rate_assumptions: FinancialGrowthAssumptions = Field(
        description="Detailed assumptions about growth rates and economic factors driving the projections."
    )

class ValidationQualityScores(BaseModel):
    """Quantitative quality assessment scores with detailed rationale"""
    score: float = Field(
        description="Quality score from 0.0 to 1.0 where 1.0 represents highest quality/confidence. Use precise decimal values (e.g., 0.85, not 0.8 or 0.9).",
        ge=0.0, 
        le=1.0
    )
    rationale: str = Field(
        description="Single, comprehensive sentence explaining the specific factors that influenced this score, including data quality, completeness, and analytical confidence."
    )

class ComprehensiveFinancialProjectionResponse(BaseModel):
    """Complete financial projection response with enhanced business intelligence and validation metrics"""
    executive_summary: str = Field(
        description="Concise 2-3 sentence summary highlighting key financial trends, growth trajectory, and critical insights from the projection analysis."
    )
    business_name: str = Field(
        description="Full legal or operating name of the business entity as identified from the financial statements."
    )
    completion_score: ValidationQualityScores = Field(
        description="Assessment of how completely all required projection elements were generated according to specifications."
    )
    data_quality_score: ValidationQualityScores = Field(
        description="Evaluation of the underlying financial data quality, completeness, consistency, and reliability for projection purposes."
    )
    projection_confidence_score: ValidationQualityScores = Field(
        description="Overall confidence level in the accuracy and reliability of the generated financial projections based on data quality and methodology."
    )
    projection_drivers_found: List[str] = Field(
        description="Specific financial metrics, ratios, or business factors that drove the projection calculations (e.g., 'Historical revenue growth rate of 15%', 'Seasonal Q4 revenue spike', 'Declining expense ratios')."
    )
    assumptions_made: List[str] = Field(
        description="Critical business and economic assumptions underlying the projections (e.g., 'Market conditions remain stable', 'No major competitive disruption', 'Inflation rate of 3% annually')."
    )
    anomalies_found: List[str] = Field(
        description="Unusual patterns, outliers, or inconsistencies discovered in the historical data that may impact projection reliability (e.g., 'Spike in expenses Q3 2023', 'Missing revenue data for 2 months')."
    )
    methodology: ProjectionMethodologyFramework = Field(
        description="Detailed documentation of the analytical approach and mathematical methods used to generate the projections."
    )
    projections_data: ComprehensiveProjectionsDataset = Field(
        description="Complete set of financial projections across all required timeframes with mathematical consistency between periods."
    )
    
    # Enhanced business intelligence fields
    key_financial_ratios: ComprehensiveFinancialRatios = Field(
        description="Critical financial health metrics calculated from projections to assess business performance and sustainability."
    )
    risk_factors: List[str] = Field(
        description="Identified financial, operational, or market risks that could materially impact the projected financial performance."
    )
    recommendations: List[str] = Field(
        description="Actionable strategic recommendations based on projection analysis to optimize financial performance and mitigate identified risks."
    )

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Financial Projection API is running"}

@app.post("/predict", response_model=ComprehensiveFinancialProjectionResponse)
async def predict(
    profit_loss_file: UploadFile = File(..., description="Profit and Loss CSV file"),
    balance_sheet_file: UploadFile = File(..., description="Balance Sheet CSV file")
):
    logger.info(f"Received request for /predict endpoint. Files: {profit_loss_file.filename}, {balance_sheet_file.filename}")
    try:
        # Validate file types
        if not profit_loss_file.filename or not profit_loss_file.filename.lower().endswith('.csv'):
            logger.warning(f"Invalid file type for profit_loss_file: {profit_loss_file.filename}")
            raise HTTPException(status_code=400, detail="Profit and Loss file must be a CSV")
        if not balance_sheet_file.filename or not balance_sheet_file.filename.lower().endswith('.csv'):
            logger.warning(f"Invalid file type for balance_sheet_file: {balance_sheet_file.filename}")
            raise HTTPException(status_code=400, detail="Balance Sheet file must be a CSV")
        
        logger.info("CSV files validated successfully. Reading contents...")
        # Read file contents
        profit_loss_content = await profit_loss_file.read()
        balance_sheet_content = await balance_sheet_file.read()
        logger.info("CSV file contents read.")
        
        prompt = """
        Use your full potential of Deep Think, reason and other relevant capabilities to analyse both the profit and loss as well as balance sheet data linked and accurately give the projections.
        
        PROJECTION REQUIREMENTS:
        Provide detailed financial projections for Revenue, Net Profit, Gross Profit, and Expenses across these timeframes (commencing January of Next Year):
        - 1 year: Monthly values (12 data points)
        - 3 years: Monthly values (36 data points) 
        - 5 years: Quarterly values (20 data points)
        - 10 years: Annual values (10 data points)
        - 15 years: Annual values (15 data points)

        RESPONSE REQUIREMENTS: 
        - Ensure all projections are realistic and defensible
        - Accuracy in the projections of Revenue, Net Profit, Gross Profit, and Expenses for all time frames are of utmost importance.
        }
        
        """
        
        logger.info("Calling Google Generative AI API...")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=profit_loss_content,
                    mime_type='text/csv',
                ),
                types.Part.from_bytes(
                    data=balance_sheet_content,
                    mime_type='text/csv',
                ),
                prompt
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ComprehensiveFinancialProjectionResponse,
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "thinking_config":{
                    "thinking_budget":32768
                }
            },
        )
        logger.info("Google Generative AI API call completed.")
        
        # Simple token usage logging
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'prompt_token_count', None) or getattr(usage, 'input_token_count', None) or 0
                output_tokens = getattr(usage, 'candidates_token_count', None) or getattr(usage, 'output_token_count', None) or 0
                total_tokens = getattr(usage, 'total_token_count', None) or 0
                
                logger.info(f"Tokens - Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total_tokens:,}")
        except Exception:
            pass  # Silently continue if token logging fails
        
        if not response.text:
            logger.error("Empty response received from AI service.")
            raise HTTPException(status_code=500, detail="Empty response from AI service")
        
        # Parse and validate the JSON response
        try:
            logger.info("Parsing and validating AI response...")
            parsed_response = json.loads(response.text)
            validated_projection = ComprehensiveFinancialProjectionResponse(**parsed_response)
            logger.info("AI response successfully parsed and validated.")
            return validated_projection
        except Exception as e:
            logger.error(f"Schema validation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Schema validation failed: {str(e)}")
            
    except HTTPException as he:
        logger.error(f"HTTP Exception occurred: {he.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"Unhandled internal server error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)