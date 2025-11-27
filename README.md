# 🌍 Real-Time AQI Monitoring and Prediction System

A comprehensive Air Quality Index (AQI) monitoring and prediction system with real-time data ingestion, machine learning models, and a modern dashboard.

![AQI Dashboard](data/dashboard_preview.png)

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [ML Model Comparison](#ml-model-comparison)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)

## ✨ Features

### Core Features
- **Real-time AQI Monitoring**: Continuous data fetching from OpenAQ API with fallback to mock data
- **Historical Data Storage**: SQLite database for storing historical AQI readings
- **ML-based Predictions**: Multiple trained classifiers for AQI category prediction
- **Interactive Dashboard**: Modern React dashboard with live updates and charts
- **Downloadable Reports**: Export data in CSV or JSON format

### Machine Learning Pipeline
- Data preprocessing with missing value handling and feature scaling
- 5 classification algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- Automatic model comparison and best model selection
- Cross-validation for robust evaluation
- Model persistence with joblib

### AQI Categories
| Category | AQI Range | Color |
|----------|-----------|-------|
| Good | 0-50 | 🟢 Green |
| Moderate | 51-100 | 🟡 Yellow |
| Unhealthy for Sensitive Groups | 101-150 | 🟠 Orange |
| Unhealthy | 151-200 | 🔴 Red |
| Very Unhealthy | 201-300 | 🟣 Purple |
| Hazardous | 301-500 | 🟤 Maroon |

## 🛠 Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **SQLAlchemy** - Database ORM
- **SQLite** - Lightweight database
- **APScheduler** - Background job scheduling
- **Pydantic** - Data validation

### Machine Learning
- **scikit-learn** - ML algorithms and evaluation
- **XGBoost** - Gradient boosting
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Recharts** - Charting library
- **Axios** - HTTP client

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Reverse proxy

## 📁 Project Structure

```
aqi-monitoring-system/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   └── aqi_model.py     # SQLAlchemy models
│   ├── ml/
│   │   └── pipeline.py      # ML training pipeline
│   ├── database/
│   │   └── connection.py    # Database configuration
│   └── utils/
│       └── helpers.py       # Utility functions
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Styles
│   ├── package.json
│   └── vite.config.js
├── data/                    # Data storage and exports
├── models/                  # Trained ML models
├── notebooks/
│   └── aqi_analysis.ipynb   # EDA and training notebook
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)

### Option 1: Local Development

#### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data models

# Run the backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Docker

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### Get Current AQI
```http
GET /api/current?location=New York
```
Response:
```json
{
  "data": [{
    "id": 1,
    "timestamp": "2024-01-15T10:30:00",
    "location": "New York",
    "aqi": 75,
    "aqi_category": "Moderate",
    "pm25": 25.5,
    "pm10": 42.0,
    "co": 1.2,
    "no2": 18.5,
    "so2": 5.0,
    "o3": 35.0
  }],
  "source": "database"
}
```

#### Get Historical Data
```http
GET /api/historical?days=7&location=Chicago
```

#### Predict AQI Category
```http
POST /api/predict
Content-Type: application/json

{
  "pm25": 35.5,
  "pm10": 50.0,
  "co": 1.5,
  "no2": 25.0,
  "so2": 8.0,
  "o3": 40.0
}
```
Response:
```json
{
  "predicted_category": "Moderate",
  "probabilities": {
    "Good": 0.15,
    "Moderate": 0.65,
    "Unhealthy for Sensitive Groups": 0.15,
    "Unhealthy": 0.03,
    "Very Unhealthy": 0.01,
    "Hazardous": 0.01
  },
  "model_used": "Random Forest"
}
```

#### Trigger Model Training
```http
POST /api/models/train
```

#### Get Model Comparison
```http
GET /api/models/comparison
```
Response:
```json
{
  "comparison": [
    {
      "model_name": "Random Forest",
      "accuracy": 0.92,
      "precision": 0.91,
      "recall": 0.92,
      "f1_score": 0.91,
      "roc_auc": 0.98,
      "is_best": true
    }
  ],
  "best_model": "Random Forest"
}
```

#### Export Data
```http
GET /api/export/csv?days=30
GET /api/export/json?days=30
```

#### Get Pollutant Trends
```http
GET /api/trends?hours=24
```

## 📊 ML Model Comparison

After training, models are compared using the following metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 92.5% | 91.8% | 92.5% | 91.7% | 0.98 |
| XGBoost | 91.8% | 90.5% | 91.8% | 90.9% | 0.97 |
| Gradient Boosting | 90.2% | 89.3% | 90.2% | 89.5% | 0.96 |
| Decision Tree | 85.4% | 84.1% | 85.4% | 84.5% | 0.91 |
| Logistic Regression | 78.6% | 77.2% | 78.6% | 77.8% | 0.92 |

*Note: Actual metrics will vary based on training data*

### Evaluation Metrics Explained
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## 🚢 Deployment

### Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3

### Railway

1. Create a new project on Railway
2. Add GitHub repository
3. Configure environment variables:
   ```
   PORT=8000
   DATABASE_URL=sqlite:///./data/aqi_data.db
   ```

### AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Clone and run
git clone https://github.com/your-repo/aqi-monitoring-system.git
cd aqi-monitoring-system
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

#### Backend
| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Backend server port | 8000 |
| `DATABASE_URL` | SQLite database path | `sqlite:///./data/aqi_data.db` |
| `FETCH_INTERVAL_MINUTES` | Data fetch interval in minutes | 15 |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000,http://localhost:5173` |

#### Frontend
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `/api` |
| `VITE_POLLING_INTERVAL` | Data polling interval in milliseconds | 60000 |
| `VITE_TRAINING_REFRESH_DELAY` | Delay to refresh after training in milliseconds | 30000 |

### Model Retraining

To retrain models with new data:

1. **Via API**:
   ```bash
   curl -X POST http://localhost:8000/api/models/train
   ```

2. **Via Dashboard**: Click "Train Models" button

3. **Via Jupyter Notebook**: Run `notebooks/aqi_analysis.ipynb`

## 🔧 Development

### Running Tests
```bash
# Backend tests (if added)
pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Structure
- Backend follows FastAPI best practices with dependency injection
- Frontend uses React hooks and functional components
- ML pipeline is modular and extensible

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

Built with ❤️ using FastAPI, React, and Machine Learning
