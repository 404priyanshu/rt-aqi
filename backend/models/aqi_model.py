"""
SQLAlchemy models for AQI data storage.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from backend.database.connection import Base


class AQIReading(Base):
    """
    Model to store AQI readings from various locations.
    """
    __tablename__ = "aqi_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    location = Column(String(255), index=True)
    city = Column(String(100))
    country = Column(String(100))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # AQI value and category
    aqi = Column(Integer)
    aqi_category = Column(String(50))
    
    # Individual pollutant readings
    pm25 = Column(Float, nullable=True)  # PM2.5
    pm10 = Column(Float, nullable=True)  # PM10
    co = Column(Float, nullable=True)    # Carbon Monoxide
    no2 = Column(Float, nullable=True)   # Nitrogen Dioxide
    so2 = Column(Float, nullable=True)   # Sulfur Dioxide
    o3 = Column(Float, nullable=True)    # Ozone
    
    # Metadata
    source = Column(String(50), default="api")
    created_at = Column(DateTime, default=func.now())
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "location": self.location,
            "city": self.city,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "aqi": self.aqi,
            "aqi_category": self.aqi_category,
            "pm25": self.pm25,
            "pm10": self.pm10,
            "co": self.co,
            "no2": self.no2,
            "so2": self.so2,
            "o3": self.o3,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ModelMetrics(Base):
    """
    Model to store ML model performance metrics.
    """
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    training_date = Column(DateTime, default=func.now())
    is_best_model = Column(Boolean, default=False)
    model_path = Column(String(255), nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "is_best_model": self.is_best_model,
            "model_path": self.model_path
        }
