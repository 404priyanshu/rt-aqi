"""
API Routes for AQI data and predictions.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import io

from backend.database.connection import get_db
from backend.models.aqi_model import AQIReading, ModelMetrics
from backend.ml.pipeline import ml_pipeline
from backend.utils.helpers import (
    generate_mock_aqi_data, 
    generate_historical_data,
    calculate_aqi_category,
    validate_aqi_reading,
    map_csv_columns
)
from pydantic import BaseModel


router = APIRouter()


# Pydantic models for request/response
class AQIReadingCreate(BaseModel):
    location: str
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    aqi: int
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    o3: Optional[float] = None


class PredictionRequest(BaseModel):
    pm25: float = 0
    pm10: float = 0
    co: float = 0
    no2: float = 0
    so2: float = 0
    o3: float = 0

class PredictionResponse(BaseModel):
    predicted_category: str
    probabilities: dict
    model_used: Optional[str] = None
    
    class Config:
        protected_namespaces = ()


# ==================== AQI Reading Endpoints ====================

@router.get("/current", summary="Get current AQI readings")
async def get_current_aqi(
    location: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get the most recent AQI readings.
    If no real data exists, returns mock data.
    """
    query = db.query(AQIReading).order_by(desc(AQIReading.timestamp))
    
    if location:
        query = query.filter(AQIReading.location.ilike(f"%{location}%"))
    
    readings = query.limit(10).all()
    
    if not readings:
        # Return mock data if no real data exists
        mock_data = generate_mock_aqi_data(location or "Default City")
        return {
            "data": [mock_data],
            "source": "mock",
            "message": "No real data available. Showing mock data."
        }
    
    return {
        "data": [r.to_dict() for r in readings],
        "source": "database",
        "message": "Real-time AQI data"
    }


@router.get("/historical", summary="Get historical AQI data")
async def get_historical_aqi(
    location: Optional[str] = None,
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get historical AQI readings for the specified period.
    """
    start_date = datetime.now() - timedelta(days=days)
    
    query = db.query(AQIReading).filter(
        AQIReading.timestamp >= start_date
    ).order_by(desc(AQIReading.timestamp))
    
    if location:
        query = query.filter(AQIReading.location.ilike(f"%{location}%"))
    
    readings = query.all()
    
    if not readings:
        # Generate historical mock data
        historical = generate_historical_data(days=days)
        return {
            "data": historical,
            "source": "mock",
            "message": f"No historical data available. Showing mock data for {days} days."
        }
    
    return {
        "data": [r.to_dict() for r in readings],
        "source": "database",
        "period_days": days
    }


@router.post("/readings", summary="Create new AQI reading")
async def create_aqi_reading(
    reading: AQIReadingCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new AQI reading in the database.
    """
    # Validate and calculate category
    aqi_category = calculate_aqi_category(reading.aqi)
    
    db_reading = AQIReading(
        location=reading.location,
        city=reading.city or reading.location,
        country=reading.country or "Unknown",
        latitude=reading.latitude,
        longitude=reading.longitude,
        aqi=reading.aqi,
        aqi_category=aqi_category,
        pm25=reading.pm25,
        pm10=reading.pm10,
        co=reading.co,
        no2=reading.no2,
        so2=reading.so2,
        o3=reading.o3,
        source="api"
    )
    
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    
    return {
        "message": "Reading created successfully",
        "data": db_reading.to_dict()
    }


@router.get("/trends", summary="Get pollutant trends")
async def get_pollutant_trends(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get pollutant trends for the last N hours.
    Returns data suitable for charting.
    """
    start_time = datetime.now() - timedelta(hours=hours)
    
    readings = db.query(AQIReading).filter(
        AQIReading.timestamp >= start_time
    ).order_by(AQIReading.timestamp).all()
    
    if not readings:
        # Generate mock trend data
        trend_data = []
        for h in range(hours):
            timestamp = datetime.now() - timedelta(hours=hours-h)
            mock = generate_mock_aqi_data("Trend City")
            mock["timestamp"] = timestamp.isoformat()
            trend_data.append(mock)
        
        return {
            "trends": trend_data,
            "source": "mock",
            "hours": hours
        }
    
    return {
        "trends": [r.to_dict() for r in readings],
        "source": "database",
        "hours": hours
    }


# ==================== Prediction Endpoints ====================

@router.post("/predict", response_model=PredictionResponse, summary="Predict AQI category")
async def predict_aqi_category(request: PredictionRequest):
    """
    Predict AQI category based on pollutant readings.
    Uses the best trained ML model.
    """
    try:
        prediction = ml_pipeline.predict({
            "pm25": request.pm25,
            "pm10": request.pm10,
            "co": request.co,
            "no2": request.no2,
            "so2": request.so2,
            "o3": request.o3
        })
        
        return PredictionResponse(
            predicted_category=prediction["predicted_category"],
            probabilities=prediction["probabilities"],
            model_used=prediction.get("model_used")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ==================== Model Endpoints ====================

@router.get("/models/comparison", summary="Get model comparison")
async def get_model_comparison():
    """
    Get comparison of all trained ML models.
    """
    comparison = ml_pipeline.get_model_comparison()
    
    if not comparison:
        return {
            "message": "No models trained yet. Trigger training first.",
            "comparison": []
        }
    
    return {
        "comparison": comparison,
        "best_model": ml_pipeline.best_model_name
    }


@router.get("/models/metrics", summary="Get model metrics from database")
async def get_stored_model_metrics(db: Session = Depends(get_db)):
    """
    Get stored model metrics from the database.
    """
    metrics = db.query(ModelMetrics).order_by(desc(ModelMetrics.training_date)).all()
    
    return {
        "metrics": [m.to_dict() for m in metrics]
    }


@router.post("/models/train", summary="Trigger model training")
async def trigger_model_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger retraining of all ML models.
    Uses historical data from the database or generates mock data.
    """
    def train_models():
        try:
            # Get training data
            readings = db.query(AQIReading).all()
            
            if len(readings) < 100:
                # Generate synthetic data for training
                print("Generating synthetic training data...")
                historical = generate_historical_data(days=90)
                df = pd.DataFrame(historical)
            else:
                df = pd.DataFrame([r.to_dict() for r in readings])
            
            # Preprocess and train
            X, y = ml_pipeline.preprocess_data(df)
            results = ml_pipeline.train_all_models(X, y)
            
            # Save models
            ml_pipeline.save_models()
            
            # Store metrics in database
            for name, metrics in results.items():
                db_metrics = ModelMetrics(
                    model_name=name,
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    roc_auc=metrics.get("roc_auc"),
                    confusion_matrix=metrics["confusion_matrix"],
                    is_best_model=(name == ml_pipeline.best_model_name)
                )
                db.add(db_metrics)
            
            db.commit()
            print("Model training completed!")
            
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    background_tasks.add_task(train_models)
    
    return {
        "message": "Model training started in background",
        "status": "training"
    }


@router.post("/models/train-from-csv", summary="Train models from CSV file")
async def train_from_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Train ML models from an uploaded CSV file.
    
    The CSV file should contain pollutant data with columns like:
    - pm2_5 or pm25 (PM2.5 readings)
    - pm10 (PM10 readings)
    - co, no, no2, o3, so2 (other pollutants)
    - AQI_Category or aqi_category (target labels)
    
    Column names are automatically mapped to the expected format.
    """
    # Validate file type by extension
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        
        # Validate content by attempting to parse as CSV
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid CSV file content. Unable to parse the file."
            )
        
        # Apply column name mapping
        df = map_csv_columns(df)
        
        # Validate minimum sample count
        min_samples = 50
        if len(df) < min_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Got {len(df)} samples, "
                       f"but minimum required is {min_samples}."
            )
        
        # Preprocess and train
        X, y = ml_pipeline.preprocess_data(df)
        results = ml_pipeline.train_all_models(X, y)
        
        # Save models
        saved_paths = ml_pipeline.save_models()
        
        # Store metrics in database
        for name, metrics in results.items():
            db_metrics = ModelMetrics(
                model_name=name,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                roc_auc=metrics.get("roc_auc"),
                confusion_matrix=metrics["confusion_matrix"],
                is_best_model=(name == ml_pipeline.best_model_name)
            )
            db.add(db_metrics)
        
        db.commit()
        
        return {
            "message": "Training completed successfully",
            "samples_used": len(df),
            "best_model": ml_pipeline.best_model_name,
            "results": results,
            "models_saved": list(saved_paths.keys())
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


# ==================== Export Endpoints ====================

@router.get("/export/csv", summary="Export data as CSV")
async def export_data_csv(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Export AQI data as CSV format.
    """
    from fastapi.responses import StreamingResponse
    import io
    
    start_date = datetime.now() - timedelta(days=days)
    readings = db.query(AQIReading).filter(
        AQIReading.timestamp >= start_date
    ).all()
    
    if not readings:
        # Use mock data
        historical = generate_historical_data(days=days)
        df = pd.DataFrame(historical)
    else:
        df = pd.DataFrame([r.to_dict() for r in readings])
    
    # Create CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=aqi_data_{days}days.csv"}
    )


@router.get("/export/json", summary="Export data as JSON")
async def export_data_json(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Export AQI data as JSON format.
    """
    start_date = datetime.now() - timedelta(days=days)
    readings = db.query(AQIReading).filter(
        AQIReading.timestamp >= start_date
    ).all()
    
    if not readings:
        # Use mock data
        data = generate_historical_data(days=days)
    else:
        data = [r.to_dict() for r in readings]
    
    return {
        "data": data,
        "export_date": datetime.now().isoformat(),
        "period_days": days,
        "record_count": len(data)
    }
