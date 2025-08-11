# Financial Projection API

## Overview
The Financial Projection API is an intelligent financial projection engine built with FastAPI, designed to generate comprehensive financial forecasts and insights using Google's Generative AI.

## Architecture and Design

### Core Technologies
*   `fastapi`: Web framework for building APIs.
*   `uvicorn`: ASGI server to run the application.
*   `google-genai`: Client library for Google's Generative AI models.
*   `pydantic`: For data validation and defining API schemas.
*   `python-multipart`: For handling file uploads.

### Application Structure
The `main.py` file contains the core logic of the application. It includes:
*   **Imports**: Necessary libraries like FastAPI, Google GenAI client, and Pydantic.
*   **Configuration**: Setup for logging and initialization of the FastAPI application.
*   **Pydantic Models**: Defines strict data structures, such as `ProjectionSchema`, for validating API requests and responses.
*   **API Endpoints**:
    *   `/`: Provides a welcome message.
    *   `/predict`: The main endpoint for financial projection, handling CSV file uploads and AI interaction.
    *   `/health`: A simple health check endpoint.

### Data Flow
```mermaid
graph TD
    A[Client] -->|POST /predict with CSV files| B(FastAPI Application)
    B --> C{Validate File Types (.csv)?}
    C -- No --> D[HTTP 400 Error]
    C -- Yes --> E[Read CSV File Contents]
    E --> F[Construct AI Prompt]
    F --> G(Google Generative AI Model - Gemini 2.5 Pro)
    G -->|Send CSVs + Prompt (with ProjectionSchema)| H(AI Generates JSON Response)
    H --> I[Receive AI JSON Response]
    I --> J{Parse & Validate JSON against ProjectionSchema}
    J -- Validation Failed --> K[HTTP 500 Error]
    J -- Validation Success --> L[Return Validated ProjectionSchema as JSON]
    L --> A
```

## Core Functionality
*   **Automated Financial Forecasting**: Generates multi-period projections (monthly, quarterly, annual).
*   **AI-Powered Analysis**: Identifies trends, drivers, assumptions, and anomalies.
*   **Structured Business Insights**: Provides executive summaries, quality scores, risks, and recommendations.
*   **Robust Data Validation**: Uses Pydantic for ensuring data integrity.

## API Endpoints
*   `/`
*   `/predict`
*   `/health`

The `/predict` endpoint is a `POST` request that accepts two `multipart/form-data` files: `profit_loss_file` and `balance_sheet_file`.

## Getting Started
To set up and run the project locally:
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the application: `uvicorn main:app --reload`