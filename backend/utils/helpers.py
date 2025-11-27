"""
Utility functions for AQI calculations and data processing.
"""

from typing import Optional, Dict, Any
import random
from datetime import datetime, timedelta


def calculate_aqi_category(aqi: int) -> str:
    """
    Determine AQI category based on EPA standards.
    
    Args:
        aqi: The AQI value (0-500)
        
    Returns:
        AQI category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_color(category: str) -> str:
    """
    Get the color code for an AQI category.
    
    Args:
        category: AQI category string
        
    Returns:
        Hex color code
    """
    colors = {
        "Good": "#00E400",
        "Moderate": "#FFFF00",
        "Unhealthy for Sensitive Groups": "#FF7E00",
        "Unhealthy": "#FF0000",
        "Very Unhealthy": "#8F3F97",
        "Hazardous": "#7E0023"
    }
    return colors.get(category, "#FFFFFF")


def generate_mock_aqi_data(location: str = "Mock City") -> Dict[str, Any]:
    """
    Generate mock AQI data for testing when API is unavailable.
    
    Args:
        location: Location name for the mock data
        
    Returns:
        Dictionary containing mock AQI data
    """
    # Generate realistic AQI values
    pm25 = round(random.uniform(5, 150), 2)
    pm10 = round(random.uniform(10, 200), 2)
    co = round(random.uniform(0.1, 10), 2)
    no2 = round(random.uniform(5, 100), 2)
    so2 = round(random.uniform(1, 50), 2)
    o3 = round(random.uniform(10, 100), 2)
    
    # Calculate AQI based on PM2.5 (simplified)
    aqi = min(500, max(0, int(pm25 * 2)))
    
    return {
        "location": location,
        "city": location,
        "country": "USA",
        "latitude": round(random.uniform(25, 48), 4),
        "longitude": round(random.uniform(-125, -70), 4),
        "aqi": aqi,
        "aqi_category": calculate_aqi_category(aqi),
        "pm25": pm25,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "timestamp": datetime.now().isoformat(),
        "source": "mock"
    }


def generate_historical_data(days: int = 30, locations: Optional[list] = None) -> list:
    """
    Generate historical mock data for ML training.
    
    Args:
        days: Number of days of historical data to generate
        locations: List of location names
        
    Returns:
        List of historical AQI readings
    """
    if locations is None:
        locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    
    historical_data = []
    current_time = datetime.now()
    
    for day in range(days):
        for location in locations:
            # Generate multiple readings per day
            for hour in range(0, 24, 6):  # Every 6 hours
                timestamp = current_time - timedelta(days=day, hours=hour)
                
                # Add some seasonal and time-based variation
                base_pm25 = 30 + 20 * (day % 7 / 7)  # Weekly variation
                base_pm25 += 10 * abs(12 - hour) / 12  # Daily variation
                
                data = generate_mock_aqi_data(location)
                data["timestamp"] = timestamp.isoformat()
                data["pm25"] = round(base_pm25 + random.uniform(-10, 10), 2)
                data["aqi"] = min(500, max(0, int(data["pm25"] * 2)))
                data["aqi_category"] = calculate_aqi_category(data["aqi"])
                
                historical_data.append(data)
    
    return historical_data


def validate_aqi_reading(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean AQI reading data.
    
    Args:
        data: Raw AQI data dictionary
        
    Returns:
        Validated and cleaned data dictionary
    """
    # Set defaults for missing values
    defaults = {
        "pm25": 0.0,
        "pm10": 0.0,
        "co": 0.0,
        "no2": 0.0,
        "so2": 0.0,
        "o3": 0.0,
        "aqi": 0,
        "source": "unknown"
    }
    
    for key, default in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default
    
    # Ensure AQI is within valid range
    data["aqi"] = max(0, min(500, int(data.get("aqi", 0))))
    
    # Recalculate category if needed
    if "aqi_category" not in data or data["aqi_category"] is None:
        data["aqi_category"] = calculate_aqi_category(data["aqi"])
    
    return data
