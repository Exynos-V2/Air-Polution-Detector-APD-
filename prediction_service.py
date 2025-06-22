import os
import joblib
import pandas as pd
from datetime import datetime

class PredictionService:
    def __init__(self, model_path='random_forest_co2_model.joblib'):
        """
        Initialize the prediction service with the path to the trained model.
        If the model file doesn't exist, it will fall back to a dummy model.
        """
        # Construct the absolute path to the model file to ensure it's always found
        _SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(_SERVICE_DIR, model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from the specified path."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Successfully loaded model from {self.model_path}")
        except FileNotFoundError:
            print(f"Model not found at {self.model_path}. Using dummy model for testing.")
            # Create a dummy model for testing
            from sklearn.ensemble import RandomForestRegressor
            import numpy as np
            
            # Create a simple dummy model that returns a random value between 0-25
            class DummyModel:
                def predict(self, X):
                    return np.random.uniform(0, 25, size=(X.shape[0],))
                    
            self.model = DummyModel()
            print("Using dummy model - please train and save a real model for production use")
    
    def predict_co2(self, sensor_data):
        """
        Predict CO2 levels based on sensor data.
        
        Args:
            sensor_data (dict): Dictionary containing sensor readings with keys:
                - 'Alcohol' (float): Alcohol level in ppm
                - 'CO' (float): CO level in ppm
                - 'NH4' (float): NH4 level in ppm
                - 'Toluen' (float): Toluen level in ppm
                - 'Acetone' (float): Acetone level in ppm
                
        Returns:
            float: Predicted CO2 level in ppm, or None if prediction fails
        """
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # Get current time features
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()  # Monday=0, Sunday=6
            
            # Sanitize incoming data to prevent errors with empty strings
            def to_float(val):
                return float(val) if val not in [None, ''] else 0.0

            # Prepare input data in the correct format expected by the model
            # The keys for sensor_data.get() must match the keys from the MQTT message payload
            input_data = pd.DataFrame({
                'Alkohol (ppm)': [to_float(sensor_data.get('Alcohol'))],
                'CO (ppm)': [to_float(sensor_data.get('CO'))],
                'NH4 (ppm)': [to_float(sensor_data.get('NH4'))],
                'Toluen (ppm)': [to_float(sensor_data.get('Toluen'))],
                'Aseton (ppm)': [to_float(sensor_data.get('Acetone'))],
                'Jam': [hour],
                'Hari': [day_of_week]
            })

            # Make prediction
            prediction = self.model.predict(input_data)
            return round(float(prediction[0]), 2)  # Return rounded prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

# Create a singleton instance of the prediction service
prediction_service = PredictionService()
