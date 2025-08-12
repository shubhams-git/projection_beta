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
        FINANCIAL PROJECTION ANALYSIS - COMPREHENSIVE GUIDELINES

        Use your full potential of Deep Think, reasoning, and analytical capabilities to perform a thorough analysis of both the profit and loss statement and balance sheet data. Generate highly accurate, mathematically sound financial projections with rigorous attention to historical patterns, seasonality, and business fundamentals.

        ═══════════════════════════════════════════════════════════════════════════════
        CRITICAL DATA ANALYSIS REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        1. HISTORICAL DATA DEEP ANALYSIS:
        - Perform comprehensive trend analysis over the entire available historical period
        - Identify and quantify seasonal patterns using decomposition techniques
        - Calculate historical growth rates (CAGR, YoY, MoM) with statistical significance testing
        - Detect and analyze cyclical patterns, outliers, and structural breaks
        - Assess data quality and identify any gaps, anomalies, or inconsistencies
        - Calculate moving averages (3, 6, 12 month) to identify underlying trends

        2. SEASONAL PATTERN IDENTIFICATION:
        - Perform seasonal decomposition to isolate seasonal, trend, and irregular components
        - Calculate seasonal indices for each month/quarter
        - Identify peak and trough periods with statistical confidence intervals
        - Assess seasonal volatility and stability over multiple years
        - Account for any shifting seasonal patterns or evolving business cycles

        3. MATHEMATICAL VALIDATION REQUIREMENTS:
        - Ensure Revenue = Gross Profit + Cost of Goods Sold (COGS)
        - Verify Net Profit = Gross Profit - Operating Expenses - Interest - Taxes
        - Maintain consistent relationships between P&L and Balance Sheet items
        - Apply compound growth calculations with precision to 4 decimal places
        - Validate that all financial ratios remain within realistic industry ranges

        ═══════════════════════════════════════════════════════════════════════════════
        PROJECTION METHODOLOGY FRAMEWORK
        ═══════════════════════════════════════════════════════════════════════════════

        1. REVENUE PROJECTIONS:
        - Base projections on weighted combination of:
            * Historical trend analysis (40% weight)
            * Seasonal patterns adjusted for growth (35% weight)
            * Industry benchmarks and economic indicators (15% weight)
            * Business-specific factors and market conditions (10% weight)
        - Apply different growth rates for different revenue streams if identifiable
        - Consider market saturation effects for long-term projections (10+ years)
        - Factor in economic cycles and potential market disruptions

        2. EXPENSE PROJECTIONS:
        - Categorize expenses into fixed, variable, and semi-variable components
        - Variable expenses: Scale with revenue using historical ratios
        - Fixed expenses: Apply inflation adjustments (typically 2-4% annually)
        - Semi-variable expenses: Use step-function modeling where applicable
        - Account for operational leverage effects as business scales
        - Consider cost optimization opportunities and efficiency improvements

        3. PROFIT MARGIN ANALYSIS:
        - Calculate historical gross margin trends and variability
        - Project margin improvements/deterioration based on:
            * Scale economies or diseconomies
            * Competitive pressures and pricing power
            * Cost inflation vs. pricing ability
            * Operational efficiency initiatives
        - Maintain margins within realistic bounds (compare to industry benchmarks)

        ═══════════════════════════════════════════════════════════════════════════════
        SPECIFIC PROJECTION REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        TIMEFRAME SPECIFICATIONS (commencing January of Next Year):
        - 1 year: Monthly values (12 data points) - High granularity with seasonal precision
        - 3 years: Monthly values (36 data points) - Medium-term strategic planning
        - 5 years: Quarterly values (20 data points) - Long-term business planning
        - 10 years: Annual values (10 data points) - Strategic horizon planning
        - 15 years: Annual values (15 data points) - Extended strategic analysis

        FOR EACH PROJECTION PERIOD, PROVIDE:
        - Revenue (with sub-components if identifiable)
        - Gross Profit (mathematically consistent with revenue and COGS)
        - Total Expenses (broken down by category where possible)
        - Net Profit (after all expenses, interest, and taxes)

        MATHEMATICAL CONSISTENCY CHECKS:
        ✓ Monthly totals must equal quarterly aggregates
        ✓ Quarterly totals must equal annual aggregates
        ✓ Growth rates must be mathematically consistent across timeframes
        ✓ Seasonal patterns must repeat logically year over year
        ✓ All financial statement relationships must remain valid

        ═══════════════════════════════════════════════════════════════════════════════
        QUALITY ASSURANCE AND VALIDATION
        ═══════════════════════════════════════════════════════════════════════════════

        1. REALISM TESTS:
        - Compare projected growth rates against industry benchmarks
        - Ensure profit margins remain within achievable ranges
        - Validate that cash flow implications are sustainable
        - Check that working capital requirements are reasonable
        - Assess debt service capability if applicable

        2. SENSITIVITY CONSIDERATIONS:
        - Consider base, optimistic, and pessimistic scenarios
        - Account for potential market disruptions or opportunities
        - Factor in competitive responses to growth initiatives
        - Consider regulatory changes or industry shifts
        - Address potential supply chain or operational constraints

        3. BUSINESS LOGIC VALIDATION:
        - Ensure projections align with business lifecycle stage
        - Consider market size limitations and competitive dynamics
        - Validate assumptionson customer acquisition and retention
        - Assess scalability of current business model
        - Factor in potential need for additional investment/capital

        ═══════════════════════════════════════════════════════════════════════════════
        STATISTICAL AND ANALYTICAL REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        1. TREND ANALYSIS:
        - Calculate linear and non-linear trend coefficients
        - Assess statistical significance of identified trends (R-squared > 0.7 preferred)
        - Identify trend acceleration/deceleration patterns
        - Apply appropriate smoothing techniques for volatile data

        2. SEASONALITY ANALYSIS:
        - Calculate seasonal indices with 95% confidence intervals
        - Test for seasonal stability over multiple periods
        - Adjust for any evolving seasonal patterns
        - Account for calendar effects and business-specific cycles

        3. FORECASTING ACCURACY:
        - Use multiple forecasting methods and ensemble results
        - Apply appropriate weights based on historical accuracy
        - Provide confidence intervals for key projections
        - Document assumptions and methodology for transparency

        ═══════════════════════════════════════════════════════════════════════════════
        INDUSTRY AND ECONOMIC CONTEXT
        ═══════════════════════════════════════════════════════════════════════════════

        1. MACROECONOMIC FACTORS:
        - Consider inflation impact on revenues and costs
        - Factor in interest rate environment effects
        - Account for economic growth/recession scenarios
        - Consider currency fluctuation impacts if applicable

        2. INDUSTRY-SPECIFIC CONSIDERATIONS:
        - Apply relevant industry growth rates and benchmarks
        - Consider industry lifecycle and maturity stage
        - Account for technological disruption potential
        - Factor in regulatory environment changes

        3. COMPETITIVE LANDSCAPE:
        - Assess market share sustainability and growth potential
        - Consider competitive response to business growth
        - Evaluate barriers to entry and competitive moats
        - Factor in potential market consolidation effects

        ═══════════════════════════════════════════════════════════════════════════════
        OUTPUT QUALITY REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        1. MATHEMATICAL PRECISION:
        - All calculations accurate to 4 decimal places minimum
        - Consistent rounding methodology throughout
        - Cross-verification of all mathematical relationships
        - Audit trail for all major calculations and assumptions

        2. LOGICAL CONSISTENCY:
        - Projections must tell a coherent business story
        - Growth assumptions must be supportable and realistic
        - All financial relationships must remain valid
        - Seasonal patterns must be logically applied

        3. PROFESSIONAL STANDARDS:
        - Meet or exceed professional forecasting standards
        - Provide clear documentation of methodology
        - Include appropriate disclaimers and assumptions
        - Ensure projections are suitable for business planning use

        ═══════════════════════════════════════════════════════════════════════════════
        FINAL VALIDATION CHECKLIST
        ═══════════════════════════════════════════════════════════════════════════════

        Before finalizing projections, verify:
        ☐ All mathematical relationships are correct
        ☐ Seasonal patterns are consistently applied
        ☐ Growth rates are realistic and sustainable
        ☐ Margin trends are justifiable and achievable
        ☐ Cash flow implications are viable
        ☐ Industry benchmarks are considered
        ☐ Economic assumptions are reasonable
        ☐ All timeframe requirements are met
        ☐ Data quality scores accurately reflect analysis
        ☐ Recommendations are actionable and specific

        CRITICAL SUCCESS FACTORS:
        - Mathematical accuracy is paramount
        - Seasonality analysis must be thorough and precise
        - All projections must be defensible with clear rationale
        - Business logic must be sound throughout all timeframes
        - Quality scores must reflect actual confidence in projections

        Generate projections that demonstrate sophisticated financial modeling capabilities while remaining grounded in historical data patterns and realistic business assumptions.
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