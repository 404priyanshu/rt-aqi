"""
Machine Learning Pipeline for AQI Category Prediction.
Includes data preprocessing, model training, evaluation, and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import joblib
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import json


class AQIMLPipeline:
    """
    Machine Learning pipeline for AQI category prediction.
    Supports multiple classifiers and automatic model selection.
    """
    
    # AQI categories for classification
    CATEGORIES = [
        "Good", "Moderate", "Unhealthy for Sensitive Groups",
        "Unhealthy", "Very Unhealthy", "Hazardous"
    ]
    
    # Feature columns for model training
    FEATURE_COLS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the ML pipeline.
        
        Args:
            models_dir: Directory to save/load trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize classifiers
        self.classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
        }
        
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model_name = None
        self.best_model = None

    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            data: DataFrame with AQI readings
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        from backend.utils.helpers import map_csv_columns
        
        # Handle missing values
        df = data.copy()
        
        # Apply column name mapping for flexibility
        df = map_csv_columns(df)
        
        # Fill missing values with median
        for col in self.FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0
        
        # Extract features
        X = df[self.FEATURE_COLS].values
        
        # Extract and encode labels
        if "aqi_category" in df.columns:
            y_raw = df["aqi_category"].values
            # Fit label encoder on actual data to ensure contiguous labels
            unique_categories = sorted(set(y_raw))
            self.label_encoder.fit(unique_categories)
            y = self.label_encoder.transform(y_raw)
        else:
            # If no category, calculate from AQI
            y = self._calculate_categories_from_aqi(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def _calculate_categories_from_aqi(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate AQI categories from AQI values.
        
        Args:
            df: DataFrame with AQI values
            
        Returns:
            Encoded category labels
        """
        from backend.utils.helpers import calculate_aqi_category
        
        if "aqi" in df.columns:
            categories = df["aqi"].apply(calculate_aqi_category).values
        else:
            # Calculate AQI from PM2.5 (simplified)
            aqi = (df["pm25"] * 2).clip(0, 500).astype(int)
            categories = aqi.apply(calculate_aqi_category).values
        
        return self.label_encoder.transform(categories)
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2,
                         min_samples: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Train all classifiers and evaluate their performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            min_samples: Minimum number of samples required for training
            
        Returns:
            Dictionary with model metrics
            
        Raises:
            ValueError: If there are not enough samples for training
        """
        import warnings
        from collections import Counter
        
        # Validate minimum sample count
        n_samples = len(y)
        if n_samples < min_samples:
            raise ValueError(
                f"Insufficient data for training. Got {n_samples} samples, "
                f"but minimum required is {min_samples}. "
                "Please provide more training data."
            )
        
        # Check class distribution for stratification
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        n_classes = len(class_counts)
        
        # Determine if we can use stratification
        # Need at least 2 samples per class for stratified split
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            warnings.warn(
                f"Some classes have only 1 sample. Falling back to non-stratified "
                f"split. Class distribution: {dict(class_counts)}",
                UserWarning
            )
            print(f"Warning: Using non-stratified split due to class imbalance")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if use_stratify else None
            )
        except ValueError as e:
            # Fallback if stratification still fails
            print(f"Stratification failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        results = {}
        
        # Determine appropriate CV folds based on data size and class distribution
        cv_folds = min(5, min_class_count) if min_class_count >= 2 else 2
        cv_folds = max(2, cv_folds)  # At least 2-fold CV
        
        for name, classifier in self.classifiers.items():
            print(f"Training {name}...")
            
            # Train model
            classifier.fit(X_train, y_train)
            self.trained_models[name] = classifier
            
            # Predict
            y_pred = classifier.predict(X_test)
            y_pred_proba = None
            
            # Get probability predictions for ROC-AUC
            if hasattr(classifier, "predict_proba"):
                y_pred_proba = classifier.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation score with appropriate folds
            try:
                cv_scores = cross_val_score(
                    classifier, X, y, cv=cv_folds, scoring='accuracy'
                )
                metrics["cv_mean"] = float(cv_scores.mean())
                metrics["cv_std"] = float(cv_scores.std())
            except ValueError as e:
                print(f"  Cross-validation failed for {name}: {e}")
                metrics["cv_mean"] = metrics["accuracy"]
                metrics["cv_std"] = 0.0
            
            results[name] = metrics
            self.model_metrics[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Select best model based on F1 score
        self._select_best_model()
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                # For multi-class, use OvR (One-vs-Rest) approach
                roc_auc = roc_auc_score(
                    y_true, y_pred_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
                metrics["roc_auc"] = float(roc_auc)
            except ValueError:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        
        return metrics
    
    def _select_best_model(self):
        """
        Select the best model based on F1 score.
        """
        if not self.model_metrics:
            return
        
        best_name = max(
            self.model_metrics.keys(),
            key=lambda x: self.model_metrics[x]["f1_score"]
        )
        
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        
        print(f"\nBest Model: {best_name}")
        print(f"F1 Score: {self.model_metrics[best_name]['f1_score']:.4f}")
    
    def save_models(self) -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Returns:
            Dictionary mapping model names to file paths
        """
        saved_paths = {}
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        saved_paths["scaler"] = scaler_path
        
        # Save label encoder
        encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
        joblib.dump(self.label_encoder, encoder_path)
        saved_paths["label_encoder"] = encoder_path
        
        # Save each trained model
        for name, model in self.trained_models.items():
            safe_name = name.lower().replace(" ", "_")
            model_path = os.path.join(self.models_dir, f"{safe_name}.joblib")
            joblib.dump(model, model_path)
            saved_paths[name] = model_path
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                "metrics": self.model_metrics,
                "best_model": self.best_model_name,
                "training_date": datetime.now().isoformat()
            }, f, indent=2)
        saved_paths["metrics"] = metrics_path
        
        # Save best model separately for easy loading
        if self.best_model is not None:
            best_path = os.path.join(self.models_dir, "best_model.joblib")
            joblib.dump(self.best_model, best_path)
            saved_paths["best_model"] = best_path
        
        print(f"Models saved to {self.models_dir}")
        return saved_paths
    
    def load_best_model(self) -> bool:
        """
        Load the best trained model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load label encoder
            encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
            
            # Load best model
            best_path = os.path.join(self.models_dir, "best_model.joblib")
            if os.path.exists(best_path):
                self.best_model = joblib.load(best_path)
            
            # Load metrics
            metrics_path = os.path.join(self.models_dir, "model_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    self.model_metrics = data.get("metrics", {})
                    self.best_model_name = data.get("best_model")
            
            return self.best_model is not None
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict AQI category for new data.
        
        Args:
            features: Dictionary with pollutant readings
            
        Returns:
            Prediction result with category and probabilities
        """
        if self.best_model is None:
            if not self.load_best_model():
                raise ValueError("No trained model available. Please train models first.")
        
        # Prepare features
        X = np.array([[
            features.get("pm25", 0),
            features.get("pm10", 0),
            features.get("co", 0),
            features.get("no2", 0),
            features.get("so2", 0),
            features.get("o3", 0)
        ]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.best_model.predict(X_scaled)[0]
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(self.best_model, "predict_proba"):
            proba = self.best_model.predict_proba(X_scaled)[0]
            for i, cat in enumerate(self.label_encoder.classes_):
                if i < len(proba):
                    probabilities[cat] = float(proba[i])
        
        return {
            "predicted_category": category,
            "probabilities": probabilities,
            "model_used": self.best_model_name
        }
    
    def get_model_comparison(self) -> List[Dict[str, Any]]:
        """
        Get comparison of all trained models.
        
        Returns:
            List of model metrics for comparison
        """
        comparison = []
        for name, metrics in self.model_metrics.items():
            comparison.append({
                "model_name": name,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc"),
                "cv_mean": metrics.get("cv_mean", 0),
                "is_best": name == self.best_model_name
            })
        
        # Sort by F1 score descending
        comparison.sort(key=lambda x: x["f1_score"], reverse=True)
        return comparison


# Singleton instance for the application
ml_pipeline = AQIMLPipeline()
