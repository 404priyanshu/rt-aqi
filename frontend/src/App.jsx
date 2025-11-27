import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';

// API Base URL
const API_BASE = import.meta.env.VITE_API_URL || '/api';

// AQI Category Colors
const AQI_COLORS = {
  'Good': '#00E400',
  'Moderate': '#FFFF00',
  'Unhealthy for Sensitive Groups': '#FF7E00',
  'Unhealthy': '#FF0000',
  'Very Unhealthy': '#8F3F97',
  'Hazardous': '#7E0023'
};

// Get CSS class for AQI category
const getAQIClass = (category) => {
  const classes = {
    'Good': 'aqi-good',
    'Moderate': 'aqi-moderate',
    'Unhealthy for Sensitive Groups': 'aqi-usg',
    'Unhealthy': 'aqi-unhealthy',
    'Very Unhealthy': 'aqi-very-unhealthy',
    'Hazardous': 'aqi-hazardous'
  };
  return classes[category] || '';
};

// AQI Display Component
const AQIDisplay = ({ data }) => {
  if (!data) return <div className="loading"><div className="spinner"></div></div>;
  
  const currentData = Array.isArray(data) ? data[0] : data;
  
  return (
    <div className="aqi-display">
      <div className={`aqi-value ${getAQIClass(currentData.aqi_category)}`}>
        {currentData.aqi}
      </div>
      <div className={`aqi-category ${getAQIClass(currentData.aqi_category)}`}>
        {currentData.aqi_category}
      </div>
      <div className="aqi-location">
        📍 {currentData.city || currentData.location}
      </div>
    </div>
  );
};

// Pollutant Display Component
const PollutantDisplay = ({ data }) => {
  if (!data) return null;
  
  const currentData = Array.isArray(data) ? data[0] : data;
  
  const pollutants = [
    { key: 'pm25', name: 'PM2.5', unit: 'µg/m³' },
    { key: 'pm10', name: 'PM10', unit: 'µg/m³' },
    { key: 'co', name: 'CO', unit: 'mg/m³' },
    { key: 'no2', name: 'NO₂', unit: 'µg/m³' },
    { key: 'so2', name: 'SO₂', unit: 'µg/m³' },
    { key: 'o3', name: 'O₃', unit: 'µg/m³' }
  ];
  
  return (
    <div className="pollutant-grid">
      {pollutants.map(p => (
        <div key={p.key} className="pollutant-item">
          <div className="pollutant-name">{p.name}</div>
          <div className="pollutant-value">
            {currentData[p.key]?.toFixed(1) || '—'}
          </div>
          <div className="pollutant-unit">{p.unit}</div>
        </div>
      ))}
    </div>
  );
};

// Trends Chart Component
const TrendsChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="loading"><p>No trend data available</p></div>;
  }
  
  // Prepare data for chart
  const chartData = data.slice(-24).map(item => ({
    time: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    AQI: item.aqi,
    PM25: item.pm25,
    PM10: item.pm10
  }));
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
        <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
        <YAxis stroke="#94a3b8" fontSize={12} />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px'
          }} 
        />
        <Legend />
        <Line type="monotone" dataKey="AQI" stroke="#00d4ff" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="PM25" stroke="#00ff88" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="PM10" stroke="#ff7eb6" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
};

// Model Comparison Table Component
const ModelComparison = ({ comparison }) => {
  if (!comparison || comparison.length === 0) {
    return <p style={{ color: '#94a3b8' }}>No model comparison data available. Train models first.</p>;
  }
  
  return (
    <table className="model-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1 Score</th>
          <th>CV Score</th>
        </tr>
      </thead>
      <tbody>
        {comparison.map((model, idx) => (
          <tr key={idx} className={model.is_best ? 'best-model' : ''}>
            <td>
              {model.model_name}
              {model.is_best && <span className="best-badge">BEST</span>}
            </td>
            <td>{(model.accuracy * 100).toFixed(2)}%</td>
            <td>{(model.precision * 100).toFixed(2)}%</td>
            <td>{(model.recall * 100).toFixed(2)}%</td>
            <td>{(model.f1_score * 100).toFixed(2)}%</td>
            <td>{model.cv_mean ? (model.cv_mean * 100).toFixed(2) + '%' : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Prediction Form Component
const PredictionForm = () => {
  const [formData, setFormData] = useState({
    pm25: '', pm10: '', co: '', no2: '', so2: '', o3: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const payload = Object.fromEntries(
        Object.entries(formData).map(([k, v]) => [k, parseFloat(v) || 0])
      );
      
      const response = await axios.post(`${API_BASE}/predict`, payload);
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      setPrediction({ error: 'Prediction failed. Models may not be trained yet.' });
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <form onSubmit={handleSubmit} className="prediction-form">
        {Object.keys(formData).map(key => (
          <div key={key} className="form-group">
            <label>{key.toUpperCase()}</label>
            <input
              type="number"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              placeholder="0.0"
              step="0.1"
            />
          </div>
        ))}
      </form>
      <button type="button" onClick={handleSubmit} className="btn btn-primary" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict AQI Category'}
      </button>
      
      {prediction && (
        <div className="prediction-result" style={{ marginTop: '20px' }}>
          {prediction.error ? (
            <p style={{ color: '#ff6b6b' }}>{prediction.error}</p>
          ) : (
            <>
              <div className={`prediction-category ${getAQIClass(prediction.predicted_category)}`}>
                {prediction.predicted_category}
              </div>
              <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                Model: {prediction.model_used || 'Unknown'}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// Main App Component
function App() {
  const [currentAQI, setCurrentAQI] = useState(null);
  const [trends, setTrends] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState(null);
  
  // Fetch current AQI data
  const fetchCurrentAQI = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/current`);
      setCurrentAQI(response.data);
    } catch (error) {
      console.error('Error fetching current AQI:', error);
    }
  }, []);
  
  // Fetch trend data
  const fetchTrends = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/trends?hours=24`);
      setTrends(response.data.trends || []);
    } catch (error) {
      console.error('Error fetching trends:', error);
    }
  }, []);
  
  // Fetch model comparison
  const fetchModelComparison = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/models/comparison`);
      setModelComparison(response.data.comparison || []);
    } catch (error) {
      console.error('Error fetching model comparison:', error);
    }
  }, []);
  
  // Trigger model training
  const triggerTraining = async () => {
    setTrainingStatus('Training started...');
    try {
      const response = await axios.post(`${API_BASE}/models/train`);
      setTrainingStatus(response.data.message);
      // Refresh comparison after a delay
      setTimeout(fetchModelComparison, 30000);
    } catch (error) {
      setTrainingStatus('Training failed: ' + error.message);
    }
  };
  
  // Download data
  const downloadData = async (format) => {
    try {
      const url = `${API_BASE}/export/${format}?days=30`;
      if (format === 'csv') {
        window.location.href = url;
      } else {
        const response = await axios.get(url);
        const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `aqi_data_30days.${format}`;
        link.click();
      }
    } catch (error) {
      console.error('Download error:', error);
    }
  };
  
  // Initial fetch and polling
  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      await Promise.all([
        fetchCurrentAQI(),
        fetchTrends(),
        fetchModelComparison()
      ]);
      setLoading(false);
    };
    
    fetchAll();
    
    // Poll for new data every 60 seconds
    const interval = setInterval(fetchCurrentAQI, 60000);
    return () => clearInterval(interval);
  }, [fetchCurrentAQI, fetchTrends, fetchModelComparison]);
  
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🌍 AQI Monitoring System</h1>
        <p>Real-Time Air Quality Index Monitoring and Prediction</p>
        <div className="status-badge status-live">LIVE</div>
      </header>
      
      {/* Main Dashboard */}
      <div className="dashboard-grid">
        {/* Current AQI Card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Current AQI</span>
            <span className="card-badge bg-good">{currentAQI?.source || 'Loading'}</span>
          </div>
          <AQIDisplay data={currentAQI?.data} />
        </div>
        
        {/* Pollutants Card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Pollutant Levels</span>
          </div>
          <PollutantDisplay data={currentAQI?.data} />
        </div>
        
        {/* Prediction Card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Predict AQI Category</span>
          </div>
          <PredictionForm />
        </div>
      </div>
      
      {/* Trends Chart */}
      <section className="charts-section">
        <h2 className="section-title">📈 24-Hour Trends</h2>
        <div className="chart-container">
          <TrendsChart data={trends} />
        </div>
      </section>
      
      {/* Model Comparison */}
      <section className="charts-section">
        <h2 className="section-title">🤖 ML Model Comparison</h2>
        <div className="card">
          <ModelComparison comparison={modelComparison} />
          {trainingStatus && (
            <p style={{ marginTop: '15px', color: '#00d4ff' }}>{trainingStatus}</p>
          )}
        </div>
      </section>
      
      {/* Actions */}
      <section className="actions-section">
        <div className="btn-group">
          <button className="btn btn-primary" onClick={triggerTraining}>
            🔄 Train Models
          </button>
          <button className="btn btn-secondary" onClick={() => downloadData('csv')}>
            📥 Download CSV
          </button>
          <button className="btn btn-secondary" onClick={() => downloadData('json')}>
            📥 Download JSON
          </button>
          <button className="btn btn-secondary" onClick={fetchCurrentAQI}>
            🔃 Refresh Data
          </button>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="footer">
        <p>AQI Monitoring System v1.0.0 | Built with FastAPI + React + ML</p>
        <p style={{ marginTop: '5px' }}>
          Data refreshes automatically every minute | Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
        </p>
      </footer>
    </div>
  );
}

export default App;
