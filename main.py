from fastapi import FastAPI, HTTPException, UploadFile, File, Query
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

app = FastAPI(title="Enhanced Financial Projection API", version="2.0.0")

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

# NEW: Goal-based projection models
class GoalSpecification(BaseModel):
    """Client-specified financial goal"""
    target_revenue: float = Field(description="Target revenue amount to achieve")
    timeframe_years: int = Field(description="Number of years to achieve the goal")
    goal_description: str = Field(description="Description of the goal")

class GoalProjectionsData(BaseModel):
    """Goal-based projections data structure - mirrors ProjectionsData but for goal achievement"""
    three_years_monthly: List[MonthlyProjection] = Field(description="36 months of goal-adjusted monthly data")
    goal_achievement_summary: str = Field(description="Summary of how the goal is achieved")
    required_adjustments: List[str] = Field(description="Key adjustments needed to achieve the goal")
    feasibility_assessment: str = Field(description="Assessment of goal feasibility based on historical performance")

class MethodologyDetails(BaseModel):
    """Details about the projection methodology used"""
    forecasting_methods_used: List[str]
    seasonal_adjustments_applied: bool
    trend_analysis_period: str
    growth_rate_assumptions: GrowthRateAssumptions

class QualityScores(BaseModel):
    score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    rationale: str = Field(description="The 1 sentence explanation for the score given")

# ENHANCED: Updated main schema to include goal-based projections
class EnhancedProjectionSchema(BaseModel):
    """Enhanced schema for comprehensive financial projections with goal-based analysis"""
    executive_summary: str
    business_name: str
    completion_score: QualityScores = Field(description="The completion score of the projection generation as per the requirements")
    data_quality_score: QualityScores = Field(description="The Score representing the quality of data after the extractions from the attached files")
    projection_confidence_score: QualityScores = Field(description="The cumulative score representing the net quality of the final projections generated")
    projection_drivers_found: List[str] = Field(description="List of all the drivers used to make the projections. If none, then specify the exact methods used to make projections.")
    assumptions_made: List[str] = Field(description="List of all the major/important assumptions used to make the projections.")
    anomalies_found: List[str] = Field(description="List of all the major/important anomalies found in the data after thorough inspection and analysis while making the projections.")
    methodology: MethodologyDetails = Field(description="Explanation of the primary projection methodologies.")
    
    # Original projections (unchanged)
    projections_data: ProjectionsData
    
    # NEW: Goal-based projections
    goal_based_projections: Optional[GoalProjectionsData] = Field(
        description="Goal-based projections when a specific target is provided",
        default=None
    )
    goal_feasibility_score: Optional[QualityScores] = Field(
        description="Score representing the feasibility of achieving the specified goal",
        default=None
    )
    
    # Additional useful fields for business analysis
    key_financial_ratios: KeyFinancialRatios
    risk_factors: List[str]
    recommendations: List[str]

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Enhanced Financial Projection API v2.0 is running"}

@app.post("/predict", response_model=EnhancedProjectionSchema)
async def predict(
    profit_loss_file: UploadFile = File(..., description="Profit and Loss file (CSV or JSON)"),
    balance_sheet_file: UploadFile = File(..., description="Balance Sheet file (CSV or JSON)"),
    goal_target_revenue: Optional[float] = Query(None, description="Optional target revenue for goal-based projections"),
    goal_timeframe_years: Optional[int] = Query(3, description="Years to achieve the revenue goal")
):
    logger.info(f"Received request for /predict endpoint. Files: {profit_loss_file.filename}, {balance_sheet_file.filename}")
    if goal_target_revenue:
        logger.info(f"Goal-based projection requested: ${goal_target_revenue:,.2f} in {goal_timeframe_years} years")
    
    try:
        # Validate file types - now supporting both CSV and JSON
        def get_file_type(filename: str) -> str:
            if not filename:
                return None
            filename_lower = filename.lower()
            if filename_lower.endswith('.csv'):
                return 'csv'
            elif filename_lower.endswith('.json'):
                return 'json'
            else:
                return None
        
        profit_loss_type = get_file_type(profit_loss_file.filename)
        balance_sheet_type = get_file_type(balance_sheet_file.filename)
        
        if not profit_loss_type:
            logger.warning(f"Invalid file type for profit_loss_file: {profit_loss_file.filename}")
            raise HTTPException(status_code=400, detail="Profit and Loss file must be CSV or JSON")
        if not balance_sheet_type:
            logger.warning(f"Invalid file type for balance_sheet_file: {balance_sheet_file.filename}")
            raise HTTPException(status_code=400, detail="Balance Sheet file must be CSV or JSON")
        
        logger.info(f"Files validated successfully. P&L: {profit_loss_type}, Balance Sheet: {balance_sheet_type}")
        
        # Read file contents
        profit_loss_content = await profit_loss_file.read()
        balance_sheet_content = await balance_sheet_file.read()
        logger.info("File contents read successfully.")
        
        # Determine MIME types based on file extensions
        profit_loss_mime_type = 'text/csv' if profit_loss_type == 'csv' else 'text/plain'
        balance_sheet_mime_type = 'text/csv' if balance_sheet_type == 'csv' else 'text/plain'

        logger.info(f"MIME types determined - P&L: {profit_loss_mime_type}, Balance Sheet: {balance_sheet_mime_type}")
        
        # Enhanced prompt with goal-based projection capabilities
        base_prompt = """
        ENHANCED FINANCIAL PROJECTION ANALYSIS - COMPREHENSIVE GUIDELINES WITH GOAL-BASED PLANNING

        Use your full potential of Deep Think, reasoning, and analytical capabilities to perform a thorough analysis of both the profit and loss statement and balance sheet data. Generate highly accurate, mathematically sound financial projections with rigorous attention to historical patterns, seasonality, and business fundamentals.

        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        CRITICAL DATA ANALYSIS REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

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

        3. COMPREHENSIVE FINANCIAL MODELING:
        - Build sophisticated forecasting models incorporating multiple variables
        - Use advanced statistical methods (ARIMA, exponential smoothing, regression analysis)
        - Cross-validate projections using multiple methodological approaches
        - Incorporate external economic indicators and industry benchmarks
        - Apply Monte Carlo simulation for risk assessment and confidence intervals

        4. BUSINESS INTELLIGENCE EXTRACTION:
        - Identify key business drivers and their correlation with financial performance
        - Analyze cost structure evolution and operating leverage effects
        - Assess working capital patterns and cash conversion cycles
        - Evaluate profitability trends and margin compression/expansion factors
        - Detect operational efficiency improvements or deteriorations

        5. PROJECTION METHODOLOGY REQUIREMENTS:
        - Generate projections for: 1-year monthly, 3-year monthly, 5-year quarterly, 10-year annual, 15-year annual
        - Ensure mathematical consistency across all time horizons
        - Apply different modeling techniques appropriate for each timeframe
        - Incorporate macroeconomic assumptions and industry outlook
        - Validate projections against industry benchmarks and peer analysis

        6. RISK ASSESSMENT AND SCENARIO ANALYSIS:
        - Identify key risk factors affecting business performance
        - Conduct sensitivity analysis on critical assumptions
        - Develop confidence intervals for all major projections
        - Assess business model sustainability and competitive positioning
        - Evaluate external threats and market disruption risks

        7. DATA FORMAT FLEXIBILITY:
        - The data may be provided in CSV or JSON format
        - For JSON data: Extract and interpret the structured fields including section, account_name, amount, period_identifier, date_as_of, etc.
        - For CSV data: Parse and analyze traditional tabular financial statements
        - Ensure consistent interpretation regardless of input format
        - Handle multi-year historical data spanning 2020-2024 or similar ranges
        - Properly aggregate and normalize data for analysis

        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        ENHANCED QUALITY CONTROL REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

        1. COMPLETION SCORE (0.0-1.0):
        - Assess how well all requested projection timeframes are generated
        - Evaluate completeness of financial statements coverage
        - Score based on mathematical consistency and logical coherence
        - Factor in depth of analysis and insight quality

        2. DATA QUALITY SCORE (0.0-1.0):
        - Evaluate completeness and consistency of input data
        - Assess data reliability and identify missing or anomalous values
        - Score based on data coverage across time periods and accounts
        - Factor in data granularity and historical depth

        3. PROJECTION CONFIDENCE SCORE (0.0-1.0):
        - Assess reliability of projection methodologies used
        - Evaluate statistical significance of identified trends
        - Score based on historical performance validation
        - Factor in uncertainty quantification and risk assessment quality

        CRITICAL BUSINESS ANALYSIS COMPONENTS:
        - Executive Summary: Comprehensive 3-4 paragraph summary of key findings and projections
        - Business Name: Extract from data or infer from context
        - Key Financial Ratios: Calculate and project critical performance metrics
        - Risk Factors: Identify 5-10 specific risks affecting projections
        - Recommendations: Provide 5-8 actionable strategic recommendations
        - Anomalies: Document all significant data anomalies or unusual patterns discovered
        - Assumptions: List all critical assumptions underlying projections
        - Methodology: Detail statistical and analytical methods employed

        """

        # NEW: Goal-based projection requirements
        goal_prompt_addition = ""
        if goal_target_revenue:
            goal_prompt_addition = f"""

        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        GOAL-BASED PROJECTION REQUIREMENTS - BACKWARD PLANNING METHODOLOGY
        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

        CLIENT GOAL SPECIFICATION:
        - Target Revenue: ${goal_target_revenue:,.2f}
        - Timeframe: {goal_timeframe_years} years
        - Goal Achievement Date: December {datetime.now().year + (goal_timeframe_years or 3)}

        BACKWARD PLANNING METHODOLOGY:
        1. GOAL ACHIEVEMENT ANALYSIS:
        - Calculate the required Compound Annual Growth Rate (CAGR) to reach ${goal_target_revenue:,.2f} from current revenue levels
        - Assess feasibility by comparing required CAGR against:
            * Historical growth capabilities of the business
            * Industry benchmarks and growth rates
            * Market size and competitive constraints
            * Operational scalability limitations

        2. MONTHLY PATHWAY CALCULATION:
        - Work backward from the ${goal_target_revenue:,.2f} target to calculate monthly revenue targets
        - Apply the same seasonal patterns identified in historical data analysis
        - Ensure month-over-month growth rates are achievable and sustainable
        - Calculate compound monthly growth rate required: ((Target/Current)^(1/(timeframe_months)) - 1)

        3. SUPPORTING METRICS CALCULATION:
        - Revenue → Calculate goal-adjusted revenue for each month
        - Gross Profit → Apply historical gross margin patterns to goal-adjusted revenue
        - Expenses → Scale expenses considering:
            * Fixed costs remain relatively stable
            * Variable costs scale with revenue using historical ratios
            * Investment requirements for growth (marketing, staff, infrastructure)
        - Net Profit → Calculate as Gross Profit minus scaled expenses

        4. FEASIBILITY ASSESSMENT FRAMEWORK:
        - Historical Performance Analysis: Compare required growth rates against past performance
        - Market Capacity Evaluation: Assess market size and competitive constraints
        - Resource Requirements: Calculate additional resources needed (capital, personnel, infrastructure)
        - Risk Factor Analysis: Identify key risks that could derail goal achievement
        - Milestone Planning: Define quarterly and annual milestones toward goal

        5. SCENARIO STRESS TESTING:
        - Test goal achievement under various market conditions
        - Assess sensitivity to key assumption changes
        - Evaluate downside protection and contingency requirements
        - Calculate probability of goal achievement based on historical performance

        3. OPTIMIZATION RECOMMENDATIONS:
        - Identify specific areas requiring operational improvements
        - Recommend investment priorities for goal achievement
        - Suggest milestone markers for tracking progress toward goal
        - Provide alternative timeline scenarios if original goal is unfeasible

        FINAL VALIDATION CHECKLIST:
        ☐ All original projection requirements are met with same quality
        ☐ Goal-based projections mathematically achieve specified target
        ☐ Both projection sets maintain internal mathematical consistency
        ☐ Feasibility assessment is realistic and evidence-based
        ☐ Recommendations are specific and actionable
        ☐ Risk factors are identified for both scenarios
        ☐ Quality scores accurately reflect confidence in both projection sets
        """

        enhanced_quality_section = """

        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        OUTPUT FORMAT AND VALIDATION REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

        MANDATORY OUTPUT STRUCTURE:
        - Must be valid JSON conforming exactly to the EnhancedProjectionSchema
        - All financial figures must be mathematically consistent
        - All quality scores must include both score (0.0-1.0) and detailed rationale
        - All projections must be realistic and based on data-driven analysis
        - Executive summary must be comprehensive yet concise (3-4 paragraphs)

        CRITICAL VALIDATION POINTS:
        ☐ JSON structure matches schema exactly
        ☐ All required fields are populated with meaningful data
        ☐ Financial projections are mathematically consistent
        ☐ Quality scores reflect actual analysis depth
        ☐ Assumptions and methodology are clearly documented
        ☐ Risk factors are specific and relevant
        ☐ Recommendations are actionable and strategic
        """

        full_prompt = base_prompt + goal_prompt_addition + enhanced_quality_section

        logger.info("Calling Google Generative AI API with enhanced goal-based capabilities and flexible file format support...")
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnhancedProjectionSchema,
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            thinking_config=types.ThinkingConfig(
                thinking_budget=32768  
            )
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=profit_loss_content,
                    mime_type=profit_loss_mime_type,
                ),
                types.Part.from_bytes(
                    data=balance_sheet_content,
                    mime_type=balance_sheet_mime_type,
                ),
                full_prompt
            ],
            config=config,
        )
        logger.info("Google Generative AI API call completed.")
        
        # Enhanced token usage logging
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'prompt_token_count', None) or getattr(usage, 'input_token_count', None) or 0
                output_tokens = getattr(usage, 'candidates_token_count', None) or getattr(usage, 'output_token_count', None) or 0
                thinking_tokens = getattr(usage, 'thoughts_token_count', None) or 0
                total_tokens = getattr(usage, 'total_token_count', None) or 0
                
                logger.info(f"Enhanced Tokens - Input: {input_tokens:,} | Output: {output_tokens:,} | Thinking: {thinking_tokens:,} | Total: {total_tokens:,}")
                if goal_target_revenue:
                    logger.info(f"Goal-based analysis overhead - Thinking tokens: {thinking_tokens:,}")
        except Exception:
            pass  # Silently continue if token logging fails
        
        if not response.text:
            logger.error("Empty response received from AI service.")
            raise HTTPException(status_code=500, detail="Empty response from AI service")
        
        # Parse and validate the JSON response
        try:
            logger.info("Parsing and validating enhanced AI response...")
            parsed_response = json.loads(response.text)
            validated_projection = EnhancedProjectionSchema(**parsed_response)
            logger.info("Enhanced AI response successfully parsed and validated.")
            
            # Log goal achievement summary if provided
            if validated_projection.goal_based_projections:
                logger.info(f"Goal-based projections generated successfully. Target: ${goal_target_revenue:,.2f}")
                
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

# FIXED: Dedicated goal-based projection endpoint with proper Query parameters
@app.post("/predict-with-goal", response_model=EnhancedProjectionSchema)
async def predict_with_goal(
    profit_loss_file: UploadFile = File(..., description="Profit and Loss file (CSV or JSON)"),
    balance_sheet_file: UploadFile = File(..., description="Balance Sheet file (CSV or JSON)"),
    target_revenue: float = Query(..., description="Target revenue amount"),
    timeframe_years: int = Query(3, description="Years to achieve goal")
):
    """
    Dedicated endpoint for goal-based financial projections.
    This endpoint requires a goal specification and focuses on backward planning methodology.
    """
    logger.info(f"Goal-specific endpoint called. Target: ${target_revenue:,.2f} in {timeframe_years} years")
    
    return await predict(
        profit_loss_file=profit_loss_file,
        balance_sheet_file=balance_sheet_file,
        goal_target_revenue=target_revenue,
        goal_timeframe_years=timeframe_years
    )

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.0.0"}

# NEW: Request model for goal calculation
class GoalRequirementsRequest(BaseModel):
    """Request model for goal calculation endpoint"""
    current_revenue: float = Field(..., description="Current annual revenue")
    target_revenue: float = Field(..., description="Target revenue amount")
    timeframe_years: int = Field(..., description="Years to achieve goal")

# NEW: Goal calculation utility endpoint
@app.post("/calculate-goal-requirements")
async def calculate_goal_requirements(request: GoalRequirementsRequest):
    """
    Utility endpoint to calculate the required growth rate for a specified goal.
    Useful for goal feasibility assessment before running full projections.
    """
    logger.info(f"Goal requirements calculation: ${request.current_revenue:,.2f} → ${request.target_revenue:,.2f} in {request.timeframe_years} years")
    
    try:
        # Calculate required CAGR
        required_cagr = ((request.target_revenue / request.current_revenue) ** (1 / request.timeframe_years)) - 1
        
        # Calculate monthly growth rate
        monthly_growth_rate = ((request.target_revenue / request.current_revenue) ** (1 / (request.timeframe_years * 12))) - 1
        
        # Calculate total growth multiple
        growth_multiple = request.target_revenue / request.current_revenue
        
        response_data = {
            "current_revenue": request.current_revenue,
            "target_revenue": request.target_revenue,
            "timeframe_years": request.timeframe_years,
            "required_cagr": round(required_cagr * 100, 2),  
            "required_monthly_growth": round(monthly_growth_rate * 100, 2),  
            "growth_multiple": round(growth_multiple, 2),
            "feasibility_assessment": "Requires detailed analysis with historical data",
            "recommendation": "Use /predict endpoint with goal parameters for comprehensive analysis"
        }
        
        logger.info(f"Goal calculation completed. Required CAGR: {required_cagr * 100:.2f}%")
        return response_data
        
    except Exception as e:
        logger.error(f"Goal calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Goal calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced Financial Projection API v2.0 with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)