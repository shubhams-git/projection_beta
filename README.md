## Enhanced Financial Projection API (v2.0.0)

### Overview
An intelligent FastAPI service that generates comprehensive financial forecasts and insights using Google's Generative AI (Gemini 2.5 Pro with thinking capabilities). Version 2.0 introduces goal-based projections and an auxiliary growth-requirements calculator.

### Architecture and Design

#### Application Structure
The `main.py` file contains the core logic:

- **Comprehensive Pydantic Models**: Strongly typed schema with strict validation
- **AI Integration**: Gemini 2.5 Pro with structured output (`response_schema`) and thinking budget
- **Robust Error Handling**: Clear HTTP errors with detailed logging
- **CSV Validation**: Strict file-type checks and content reading
- **Usage Tracking**: Logs input/output/thinking token usage when available

#### Pydantic Schema Hierarchy
```
EnhancedProjectionSchema
├── executive_summary: str
├── business_name: str
├── completion_score: QualityScores
├── data_quality_score: QualityScores
├── projection_confidence_score: QualityScores
├── projection_drivers_found: List[str]
├── assumptions_made: List[str]
├── anomalies_found: List[str]
├── methodology: MethodologyDetails
├── projections_data: ProjectionsData
│   ├── one_year_monthly: List[MonthlyProjection]        # 12
│   ├── three_years_monthly: List[MonthlyProjection]     # 36
│   ├── five_years_quarterly: List[QuarterlyProjection]  # 20
│   ├── ten_years_annual: List[AnnualProjection]         # 10
│   └── fifteen_years_annual: List[AnnualProjection]     # 15
├── goal_based_projections?: GoalProjectionsData
│   ├── three_years_monthly: List[MonthlyProjection]
│   ├── goal_achievement_summary: str
│   ├── required_adjustments: List[str]
│   └── feasibility_assessment: str
├── goal_feasibility_score?: QualityScores
├── key_financial_ratios: KeyFinancialRatios
├── risk_factors: List[str]
└── recommendations: List[str]

QualityScores
├── score: float (0.0-1.0)
└── rationale: str

MethodologyDetails
├── forecasting_methods_used: List[str]
├── seasonal_adjustments_applied: bool
├── trend_analysis_period: str
└── growth_rate_assumptions: GrowthRateAssumptions

GrowthRateAssumptions
├── revenue_cagr: float
├── expense_inflation: float
└── profit_margin_target: float

MonthlyProjection: { month, revenue, net_profit, gross_profit, expenses }
QuarterlyProjection: { quarter, revenue, net_profit, gross_profit, expenses }
AnnualProjection: { year, revenue, net_profit, gross_profit, expenses }

GoalProjectionsData
├── three_years_monthly: List[MonthlyProjection]
├── goal_achievement_summary: str
├── required_adjustments: List[str]
└── feasibility_assessment: str
```

#### Data Flow
```mermaid
graph TD
    A[Client] -->|POST /predict with CSVs (+ optional goal params)| B(FastAPI)
    A -->|POST /predict-with-goal with CSVs + required goal params| B
    A -->|POST /calculate-goal-requirements (JSON)| C[Utility Calculator]
    B --> D{Validate File Types}
    D -- No --> E[HTTP 400]
    D -- Yes --> F[Read CSV File Contents]
    F --> G[Construct Enhanced Prompt]
    G --> H(Google GenAI - Gemini 2.5 Pro)
    H --> I[AI JSON Response]
    I --> J{Parse & Validate against EnhancedProjectionSchema}
    J -- Fail --> K[HTTP 500]
    J -- Success --> L[Return Validated JSON]
    L --> A
    C --> M[Compute CAGR & Monthly Growth]
    M --> N[Return JSON]
    N --> A
```

## Core Functionality

### Projections Generated
- **1 Year Monthly**: 12 monthly projections (starting next January)
- **3 Years Monthly**: 36 monthly projections
- **5 Years Quarterly**: 20 quarterly projections
- **10 Years Annual**: 10 annual projections
- **15 Years Annual**: 15 annual projections

### Goal-Based Planning (New)
- Optional inputs allow backward planning to a target revenue over a timeframe
- Generates a dedicated 36-month pathway, feasibility assessment, and required adjustments

### Business Intelligence
- **Executive Summary** and methodology details
- **Quality Scores**: completion, data quality, projection confidence
- **Drivers, Assumptions, Anomalies** explicitly listed
- **Financial Ratios**, **Risk Factors**, and **Recommendations**

### AI Configuration
- **Model**: `gemini-2.5-pro`
- **Structured Output**: `response_schema=EnhancedProjectionSchema` (JSON)
- **Thinking**: `thinking_budget=32768`
- **Sampling**: `temperature=0.1`, `top_p=0.8`, `top_k=40`
- **Inputs**: Two CSV files (P&L and Balance Sheet) via multipart form data

## API Endpoints

### `GET /`
Returns: `{ "message": "Enhanced Financial Projection API v2.0 is running" }`

### `POST /predict`
Main endpoint for projections. Accepts optional goal parameters.

- **Request**: `multipart/form-data`
  - `profit_loss_file`: CSV (required)
  - `balance_sheet_file`: CSV (required)
  - Query (optional): `goal_target_revenue: float`, `goal_timeframe_years: int` (default 3)
- **Response**: `EnhancedProjectionSchema` JSON

### `POST /predict-with-goal`
Dedicated goal-based endpoint; requires goal parameters.

- **Request**: `multipart/form-data` + Query
  - `profit_loss_file`: CSV (required)
  - `balance_sheet_file`: CSV (required)
  - `target_revenue: float` (required)
  - `timeframe_years: int` (default 3)
- **Response**: `EnhancedProjectionSchema` JSON

### `POST /calculate-goal-requirements`
Utility endpoint to compute required CAGR and monthly growth.

- **Request (JSON)**:
  ```json
  { "current_revenue": 1000000, "target_revenue": 5000000, "timeframe_years": 3 }
  ```
- **Response (JSON)**:
  ```json
  {
    "current_revenue": 1000000,
    "target_revenue": 5000000,
    "timeframe_years": 3,
    "required_cagr": 73.21,
    "required_monthly_growth": 4.62,
    "growth_multiple": 5.0,
    "feasibility_assessment": "Requires detailed analysis with historical data",
    "recommendation": "Use /predict endpoint with goal parameters for comprehensive analysis"
  }
  ```

### `GET /health`
Returns: `{ "status": "healthy", "timestamp": "...", "version": "2.0.0" }`

### Interactive API Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing with Postman

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Set `GOOGLE_API_KEY` in your environment
   - macOS/Linux: `export GOOGLE_API_KEY="YOUR_API_KEY"`
   - Windows PowerShell (current session): `$env:GOOGLE_API_KEY="YOUR_API_KEY"`
   - Windows (persist): `setx GOOGLE_API_KEY "YOUR_API_KEY"`
3. Start server: `python main.py`
4. Verify server: `http://localhost:8000`

### Step-by-Step Tests

#### 1) Health Check
- Method: `GET`
- URL: `http://localhost:8000/health`
- Expect: `{ "status": "healthy", ... }`

#### 2) Root
- Method: `GET`
- URL: `http://localhost:8000/`
- Expect: `{ "message": "Enhanced Financial Projection API v2.0 is running" }`

#### 3) Projections (Main)
- Method: `POST`
- URL: `http://localhost:8000/predict`
- Body: `form-data` with two CSV files
- Optional query: `goal_target_revenue`, `goal_timeframe_years`

#### 4) Goal-Based Projections (Dedicated)
- Method: `POST`
- URL: `http://localhost:8000/predict-with-goal?target_revenue=5000000&timeframe_years=3`
- Body: `form-data` with two CSV files

#### 5) Goal Requirements Utility
- Method: `POST`
- URL: `http://localhost:8000/calculate-goal-requirements`
- Body (JSON): `{ "current_revenue": 1000000, "target_revenue": 5000000, "timeframe_years": 3 }`

## cURL Examples

### Health
```bash
curl -s http://localhost:8000/health
```

### Run Projections (baseline)
```bash
curl -X POST \
  -F profit_loss_file=@path/to/profit_loss.csv \
  -F balance_sheet_file=@path/to/balance_sheet.csv \
  http://localhost:8000/predict
```

### Run Projections with goal (optional query)
```bash
curl -X POST \
  -F profit_loss_file=@path/to/profit_loss.csv \
  -F balance_sheet_file=@path/to/balance_sheet.csv \
  "http://localhost:8000/predict?goal_target_revenue=5000000&goal_timeframe_years=3"
```

### Dedicated Goal Endpoint
```bash
curl -X POST \
  -F profit_loss_file=@path/to/profit_loss.csv \
  -F balance_sheet_file=@path/to/balance_sheet.csv \
  "http://localhost:8000/predict-with-goal?target_revenue=5000000&timeframe_years=3"
```

### Goal Requirements Calculator
```bash
curl -X POST http://localhost:8000/calculate-goal-requirements \
  -H "Content-Type: application/json" \
  -d '{"current_revenue":1000000,"target_revenue":5000000,"timeframe_years":3}'
```

## Sample Response Structure (EnhancedProjectionSchema)
```json
{
  "executive_summary": "Business shows strong growth potential...",
  "business_name": "Example Co.",
  "completion_score": { "score": 0.95, "rationale": "..." },
  "data_quality_score": { "score": 0.87, "rationale": "..." },
  "projection_confidence_score": { "score": 0.89, "rationale": "..." },
  "projection_drivers_found": ["seasonality", "trend", "benchmarks"],
  "assumptions_made": ["inflation at 3%", "stable margins"],
  "anomalies_found": ["one-off expense in 2023-07"],
  "methodology": {
    "forecasting_methods_used": ["trend", "seasonal decomposition"],
    "seasonal_adjustments_applied": true,
    "trend_analysis_period": "36 months",
    "growth_rate_assumptions": {
      "revenue_cagr": 0.18,
      "expense_inflation": 0.03,
      "profit_margin_target": 0.22
    }
  },
  "projections_data": {
    "one_year_monthly": [ { "month": "2026-01", "revenue": 0, "gross_profit": 0, "expenses": 0, "net_profit": 0 } ],
    "three_years_monthly": [],
    "five_years_quarterly": [],
    "ten_years_annual": [],
    "fifteen_years_annual": []
  },
  "goal_based_projections": {
    "three_years_monthly": [],
    "goal_achievement_summary": "...",
    "required_adjustments": ["increase marketing", "hire sales"],
    "feasibility_assessment": "..."
  },
  "goal_feasibility_score": { "score": 0.72, "rationale": "..." },
  "key_financial_ratios": { "gross_margin": 0.45, "net_margin": 0.12, "current_ratio": 1.8, "debt_to_equity": 0.6 },
  "risk_factors": ["market slowdown"],
  "recommendations": ["optimize COGS", "phase hiring"]
}
```

## Performance Considerations
- Typical response time: 30–60 seconds for complex analysis
- CSV size recommendation: < 10MB each
- Token usage (input/output/thinking) logged when available
- FastAPI handles concurrent requests

## Production Deployment
- Run: `python main.py` (Uvicorn inside)
- Host: `0.0.0.0`, Port: `8000`
- Env: `GOOGLE_API_KEY` must be set
- Monitoring: structured logging; `/health` for liveness

## Error Codes
- `400`: Invalid file format
- `422`: Request validation failed
- `500`: Internal server error or AI service unavailable