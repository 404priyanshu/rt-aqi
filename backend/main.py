"""
Main FastAPI Application for AQI Monitoring System.
Includes API endpoints, scheduler for data fetching, and ML integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import httpx
import os
from datetime import datetime

from backend.database.connection import init_db, SessionLocal
from backend.api.routes import router as aqi_router
from backend.models.aqi_model import AQIReading
from backend.utils.helpers import generate_mock_aqi_data, calculate_aqi_category
from backend.ml.pipeline import ml_pipeline


# Scheduler instance
scheduler = AsyncIOScheduler()


async def fetch_aqi_data():
    """
    Scheduled task to fetch AQI data from external API or generate mock data.
    Called periodically by the scheduler.
    """
    print(f"[{datetime.now()}] Fetching AQI data...")
    
    # List of cities to monitor
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    
    db = SessionLocal()
    try:
        for city in cities:
            try:
                # Try to fetch from OpenAQ API (free, no API key required)
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"https://api.openaq.org/v2/latest?city={city}&limit=1",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("results"):
                            result = data["results"][0]
                            measurements = {m["parameter"]: m["value"] for m in result.get("measurements", [])}
                            
                            # Calculate AQI from PM2.5 (simplified)
                            pm25 = measurements.get("pm25", 0)
                            aqi = min(500, max(0, int(pm25 * 2)))
                            
                            reading = AQIReading(
                                location=result.get("location", city),
                                city=result.get("city", city),
                                country=result.get("country", "USA"),
                                latitude=result.get("coordinates", {}).get("latitude"),
                                longitude=result.get("coordinates", {}).get("longitude"),
                                aqi=aqi,
                                aqi_category=calculate_aqi_category(aqi),
                                pm25=measurements.get("pm25"),
                                pm10=measurements.get("pm10"),
                                co=measurements.get("co"),
                                no2=measurements.get("no2"),
                                so2=measurements.get("so2"),
                                o3=measurements.get("o3"),
                                source="openaq"
                            )
                            db.add(reading)
                            continue
                            
            except Exception as e:
                print(f"API fetch failed for {city}: {e}")
            
            # Fallback to mock data
            mock_data = generate_mock_aqi_data(city)
            reading = AQIReading(
                location=mock_data["location"],
                city=mock_data["city"],
                country=mock_data["country"],
                latitude=mock_data["latitude"],
                longitude=mock_data["longitude"],
                aqi=mock_data["aqi"],
                aqi_category=mock_data["aqi_category"],
                pm25=mock_data["pm25"],
                pm10=mock_data["pm10"],
                co=mock_data["co"],
                no2=mock_data["no2"],
                so2=mock_data["so2"],
                o3=mock_data["o3"],
                source="mock"
            )
            db.add(reading)
        
        db.commit()
        print(f"[{datetime.now()}] Data fetch completed for {len(cities)} cities")
        
    except Exception as e:
        print(f"Error in scheduled fetch: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes database and starts the scheduler on startup.
    Shuts down scheduler on application shutdown.
    """
    # Startup
    print("Initializing AQI Monitoring System...")
    
    # Initialize database
    init_db()
    print("Database initialized")
    
    # Try to load existing ML model
    if ml_pipeline.load_best_model():
        print(f"Loaded ML model: {ml_pipeline.best_model_name}")
    else:
        print("No pre-trained model found. Train models using /api/models/train endpoint.")
    
    # Start scheduler
    scheduler.add_job(
        fetch_aqi_data,
        trigger=IntervalTrigger(minutes=int(os.getenv("FETCH_INTERVAL_MINUTES", "15"))),
        id="aqi_data_fetch",
        name="Fetch AQI Data",
        replace_existing=True
    )
    scheduler.start()
    print("Scheduler started - fetching data every 15 minutes")
    
    # Initial data fetch
    await fetch_aqi_data()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    scheduler.shutdown()


# Create FastAPI application
app = FastAPI(
    title="AQI Monitoring System",
    description="Real-Time Air Quality Index Monitoring and Prediction System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(aqi_router, prefix="/api", tags=["AQI"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint returning API information.
    """
    return {
        "name": "AQI Monitoring System API",
        "version": "1.0.0",
        "description": "Real-Time Air Quality Index Monitoring and Prediction System",
        "endpoints": {
            "current_aqi": "/api/current",
            "historical": "/api/historical",
            "predict": "/api/predict",
            "model_comparison": "/api/models/comparison",
            "train_models": "/api/models/train",
            "export_csv": "/api/export/csv",
            "export_json": "/api/export/json",
            "docs": "/docs"
        }
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "scheduler_running": scheduler.running,
        "model_loaded": ml_pipeline.best_model is not None
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
