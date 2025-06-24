import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedPredictionService:
    def __init__(self, model_path='enhanced_co2_model.joblib'):
        """
        Initialize the enhanced prediction service.
        This service can handle both the old simple model and new enhanced models.
        """
        self.model_path = self._get_model_path(model_path)
        self.model_package = None
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.use_scaling = False
        self.is_enhanced_model = False
        self.load_model()
    
    def _get_model_path(self, model_path):
        """Get the absolute path to the model file"""
        if os.path.isabs(model_path):
            return model_path
        
        # Try current directory first
        if os.path.exists(model_path):
            return os.path.abspath(model_path)
        
        # Try in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, model_path)
        if os.path.exists(full_path):
            return full_path
        
        # Fallback to original model
        original_model_path = os.path.join(script_dir, 'random_forest_co2_model.joblib')
        if os.path.exists(original_model_path):
            print(f"Enhanced model not found, falling back to: {original_model_path}")
            return original_model_path
        
        return model_path  # Return as-is if nothing found
    
    def load_model(self):
        """Load the model (enhanced or legacy)"""
        try:
            loaded_data = joblib.load(self.model_path)
            
            # Check if this is an enhanced model package
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                print(f"Loading enhanced model from {self.model_path}")
                self.model_package = loaded_data
                self.model = loaded_data['model']
                self.scaler = loaded_data.get('scaler', None)
                self.feature_names = loaded_data.get('feature_names', [])
                self.use_scaling = loaded_data.get('use_scaling', False)
                self.is_enhanced_model = True
                print(f"Enhanced model features: {len(self.feature_names)}")
                
            else:
                # Legacy simple model
                print(f"Loading legacy model from {self.model_path}")
                self.model = loaded_data
                self.is_enhanced_model = False
                print("Using legacy model - consider upgrading to enhanced model")
                
        except FileNotFoundError:
            print(f"Model not found at {self.model_path}. Using dummy model.")
            self._create_dummy_model()
        except Exception as e:
            print(f"Error loading model: {e}. Using dummy model.")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing"""
        from sklearn.ensemble import RandomForestRegressor
        
        class DummyModel:
            def predict(self, X):
                return np.random.uniform(5, 25, size=(X.shape[0],))
                
        self.model = DummyModel()
        self.is_enhanced_model = False
        print("Using dummy model - please train a real model for production")
    
    def _engineer_features_legacy(self, sensor_data):
        """Create features for legacy model (simple approach)"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        def to_float(val):
            try:
                return float(val) if val not in [None, '', 'null'] else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        # Create DataFrame in the exact order expected by legacy model
        input_data = pd.DataFrame({
            'Alkohol (ppm)': [to_float(sensor_data.get('Alcohol'))],
            'CO (ppm)': [to_float(sensor_data.get('CO'))],
            'NH4 (ppm)': [to_float(sensor_data.get('NH4'))],
            'Toluen (ppm)': [to_float(sensor_data.get('Toluen'))],
            'Aseton (ppm)': [to_float(sensor_data.get('Acetone'))],
            'Jam': [hour],
            'Hari': [day_of_week]
        })
        
        return input_data
    
    def _engineer_features_enhanced(self, sensor_data):
        """Create enhanced features for the new model"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        month = now.month
        
        def to_float(val):
            try:
                return float(val) if val not in [None, '', 'null'] else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        # Get basic sensor values
        alkohol = to_float(sensor_data.get('Alcohol'))
        co = to_float(sensor_data.get('CO'))
        nh4 = to_float(sensor_data.get('NH4'))
        toluen = to_float(sensor_data.get('Toluen'))
        acetone = to_float(sensor_data.get('Acetone'))
        
        # Create enhanced features
        features = {
            # Basic sensor readings
            'Alkohol (ppm)': alkohol,
            'CO (ppm)': co,
            'NH4 (ppm)': nh4,
            'Toluen (ppm)': toluen,
            'Aseton (ppm)': acetone,
            
            # Time features
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Month': month,
            'IsWeekend': int(day_of_week >= 5),
            
            # Cyclical time features
            'Hour_sin': np.sin(2 * np.pi * hour / 24),
            'Hour_cos': np.cos(2 * np.pi * hour / 24),
            'DayOfWeek_sin': np.sin(2 * np.pi * day_of_week / 7),
            'DayOfWeek_cos': np.cos(2 * np.pi * day_of_week / 7),
            'Month_sin': np.sin(2 * np.pi * month / 12),
            'Month_cos': np.cos(2 * np.pi * month / 12),
            
            # Interaction features
            'CO_Alcohol_interaction': co * alkohol,
            'Toluen_Acetone_interaction': toluen * acetone,
            
            # Composite features
            'Total_VOCs': alkohol + toluen + acetone,
            'Combustion_gases': co + nh4,
            
            # Time-based indicators
            'Morning_rush': int(7 <= hour <= 9),
            'Evening_rush': int(17 <= hour <= 19),
            'Night_time': int(hour >= 22 or hour <= 6),
        }
        
        # Create DataFrame with features in the correct order
        if self.feature_names:
            # Use the exact feature order from training
            ordered_features = {}
            for feature_name in self.feature_names:
                if feature_name in features:
                    ordered_features[feature_name] = features[feature_name]
                else:
                    # Handle missing features gracefully
                    print(f"Missing feature: {feature_name}, using default value 0")
                    ordered_features[feature_name] = 0.0
            
            input_data = pd.DataFrame([ordered_features])
        else:
            # Fallback if feature names not available
            input_data = pd.DataFrame([features])
        
        return input_data
    
    def predict_co2(self, sensor_data):
        """
        Predict CO2 levels based on sensor data.
        
        Args:
            sensor_data (dict): Dictionary containing sensor readings
                
        Returns:
            float: Predicted CO2 level in ppm, or None if prediction fails
        """
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # Step 1: Engineer features into a DataFrame
            if self.is_enhanced_model:
                input_df = self._engineer_features_enhanced(sensor_data)
            else:
                input_df = self._engineer_features_legacy(sensor_data)
            
            prediction_input = input_df
            
            # Step 2: Apply scaling ONLY if the model requires it
            # This makes the service compatible with future models that might use scaling
            if self.use_scaling and self.scaler is not None:
                scaled_values = self.scaler.transform(input_df)
                # Recreate DataFrame to preserve feature names for the model
                prediction_input = pd.DataFrame(scaled_values, columns=input_df.columns)

            # Step 3: Make the prediction using the prepared DataFrame
            prediction = self.model.predict(prediction_input)
            
            # Step 4: Process and return the result
            pred_value = float(prediction[0])
            pred_value = max(1.0, min(pred_value, 50.0))  # Clamp between 1-50 ppm
            
            return round(pred_value, 2)
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            print(f"Sensor data received: {sensor_data}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model_package and 'model_info' in self.model_package:
            return {
                'type': 'enhanced',
                'features': len(self.feature_names),
                'uses_scaling': self.use_scaling,
                'model_info': self.model_package['model_info']
            }
        else:
            return {
                'type': 'legacy' if self.model else 'dummy',
                'features': 7 if not self.is_enhanced_model else 0,
                'uses_scaling': False
            }
    
    def test_prediction(self):
        """Test the prediction service with sample data"""
        print("Testing prediction service...")
        
        # Sample sensor data
        test_data = {
            'Alcohol': 0.25,
            'CO': 5.5,
            'NH4': 0.30,
            'Toluen': 0.15,
            'Acetone': 0.20
        }
        
        prediction = self.predict_co2(test_data)
        
        if prediction is not None:
            print(f"Test prediction successful: {prediction} ppm CO2")
            print(f"Model info: {self.get_model_info()}")
            return True
        else:
            print("Test prediction failed")
            return False

# Create a singleton instance
try:
    enhanced_prediction_service = EnhancedPredictionService()
    print("Enhanced Prediction Service initialized successfully")
except Exception as e:
    print(f"Failed to initialize Enhanced Prediction Service: {e}")
    # Fallback to basic prediction service
    from prediction_service import prediction_service
    enhanced_prediction_service = prediction_service
    print("Falling back to basic prediction service")

if __name__ == "__main__":
    # Test the service
    service = EnhancedPredictionService()
    service.test_prediction() 