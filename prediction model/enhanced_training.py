import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our dataset generator
from dataset_generator import AirPollutionDatasetGenerator

class EnhancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_or_load_data(self, use_existing=True, dataset_size=3000):
        """Generate new dataset or load existing one"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to load enhanced dataset first
        enhanced_path = os.path.join(script_dir, 'enhanced_pollution_dataset.csv')
        original_path = os.path.join(script_dir, 'dataset_ai_1000.csv')
        
        if use_existing and os.path.exists(enhanced_path):
            print(f"Loading existing enhanced dataset from {enhanced_path}")
            df = pd.read_csv(enhanced_path)
        elif use_existing and os.path.exists(original_path):
            print(f"Loading original dataset from {original_path}")
            df = pd.read_csv(original_path)
        else:
            print(f"Generating new enhanced dataset with {dataset_size} records...")
            generator = AirPollutionDatasetGenerator()
            df = generator.generate_dataset(num_records=dataset_size)
            generator.save_dataset(df, enhanced_path)
            
        return df
    
    def feature_engineering(self, df):
        """Enhanced feature engineering"""
        print("Performing feature engineering...")
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract time features
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Month'] = df['Timestamp'].dt.month
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Create time-based cyclical features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Create interaction features
        df['CO_Alcohol_interaction'] = df['CO (ppm)'] * df['Alkohol (ppm)']
        df['Toluen_Acetone_interaction'] = df['Toluen (ppm)'] * df['Aseton (ppm)']
        
        # Create pollution composite scores
        df['Total_VOCs'] = df['Alkohol (ppm)'] + df['Toluen (ppm)'] + df['Aseton (ppm)']
        df['Combustion_gases'] = df['CO (ppm)'] + df['NH4 (ppm)']
        
        # Rush hour indicators
        df['Morning_rush'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)).astype(int)
        df['Evening_rush'] = ((df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
        df['Night_time'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        
        # Remove timestamp column
        df = df.drop('Timestamp', axis=1)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Define target and features
        target = 'CO2 (ppm)'
        
        # Separate features and target
        X = df.drop(target, axis=1)
        y = df[target]
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def save_best_model(self, filename='enhanced_co2_model.joblib'):
        """Save the best performing model"""
        if self.best_model is None:
            print("No model has been trained yet!")
            return
        
        # Create model package
        model_package = {
            'model': self.best_model['model'],
            'scaler': self.scaler if self.best_model['use_scaling'] else None,
            'feature_names': self.feature_names,
            'use_scaling': self.best_model['use_scaling'],
            'model_info': {
                'best_params': self.best_model['best_params'],
                'cv_score': self.best_model['cv_score']
            }
        }
        
        # Save to parent directory (same level as prediction_service.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(script_dir), filename)
        
        joblib.dump(model_package, model_path)
        print(f"Best model saved as: {model_path}")
        
        return model_path

def main():
    """Main training pipeline"""
    print("Starting Enhanced Air Pollution Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Load or generate data
    df = trainer.generate_or_load_data(use_existing=True, dataset_size=3000)
    
    # Feature engineering
    df_engineered = trainer.feature_engineering(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = trainer.prepare_data(df_engineered)
    
    # Simple Random Forest training for compatibility
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"   MAE: {mae:.3f} ppm")
    print(f"   R²:  {r2:.3f}")
    
    # Store best model
    trainer.best_model = {
        'model': rf_model,
        'use_scaling': False,
        'best_params': {'n_estimators': 200, 'max_depth': 15},
        'cv_score': -mae
    }
    
    # Save model
    model_path = trainer.save_best_model('enhanced_co2_model.joblib')
    
    print("\nTraining completed successfully!")
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main() 