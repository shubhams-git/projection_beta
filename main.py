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

class QualityScores(BaseModel):
    score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    rationale: str = Field(description="The 1 sentence explanation for the score given")

class ProjectionSchema(BaseModel):
    """Enhanced schema for comprehensive financial projections"""
    executive_summary: str
    business_name: str
    completion_score: QualityScores = Field(description="The completion score of the projection generation as per the requirements")
    data_quality_score: QualityScores = Field(description="The Score representing the quality of data after the extractions from the attached files")
    projection_confidence_score: QualityScores = Field(description="The cumulative score representing the net quality of the final projections generated")
    projection_drivers_found: List[str] = Field(description="List of all the drivers used to make the projections. If none, then specify the exact methods used to make projections.")
    assumptions_made: List[str] = Field(description="List of all the major/important assumptions used to make the projections.")
    anomalies_found: List[str] = Field(description="List of all the major/important anomalies found in the data after thorough inspection and analysis while making the projections.")
    methodology: MethodologyDetails = Field(description="Explanation of the primary projection methodologies.")
    projections_data: ProjectionsData
    
    # Additional useful fields for business analysis
    key_financial_ratios: KeyFinancialRatios
    risk_factors: List[str]
    recommendations: List[str]

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Financial Projection API is running"}

@app.post("/predict", response_model=ProjectionSchema)
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
                "response_schema": ProjectionSchema,
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
            validated_projection = ProjectionSchema(**parsed_response)
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