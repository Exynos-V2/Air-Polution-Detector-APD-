import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class AirPollutionDatasetGenerator:
    def __init__(self):
        """Initialize the dataset generator with realistic pollution patterns"""
        
        # Base pollution levels for different environments
        self.environment_types = {
            'urban_heavy': {
                'base_co2': 22.0,
                'base_co': 8.5,
                'base_alcohol': 0.35,
                'base_nh4': 0.40,
                'base_toluen': 0.18,
                'base_acetone': 0.28,
                'variability': 0.25
            },
            'urban_moderate': {
                'base_co2': 18.0,
                'base_co': 6.0,
                'base_alcohol': 0.25,
                'base_nh4': 0.30,
                'base_toluen': 0.12,
                'base_acetone': 0.20,
                'variability': 0.20
            },
            'suburban': {
                'base_co2': 12.0,
                'base_co': 3.5,
                'base_alcohol': 0.15,
                'base_nh4': 0.20,
                'base_toluen': 0.08,
                'base_acetone': 0.12,
                'variability': 0.15
            },
            'rural': {
                'base_co2': 8.0,
                'base_co': 2.0,
                'base_alcohol': 0.08,
                'base_nh4': 0.12,
                'base_toluen': 0.04,
                'base_acetone': 0.06,
                'variability': 0.10
            },
            'industrial': {
                'base_co2': 25.0,
                'base_co': 12.0,
                'base_alcohol': 0.45,
                'base_nh4': 0.50,
                'base_toluen': 0.25,
                'base_acetone': 0.35,
                'variability': 0.30
            }
        }
        
        # Traffic patterns (multipliers by hour)
        self.traffic_pattern = {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.5,
            6: 0.8, 7: 1.3, 8: 1.5, 9: 1.2, 10: 1.0, 11: 1.1,
            12: 1.2, 13: 1.1, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.4,
            18: 1.6, 19: 1.3, 20: 1.0, 21: 0.8, 22: 0.6, 23: 0.4
        }
        
        # Day of week patterns (multipliers)
        self.day_pattern = {
            0: 1.2,  # Monday
            1: 1.3,  # Tuesday  
            2: 1.3,  # Wednesday
            3: 1.2,  # Thursday
            4: 1.4,  # Friday
            5: 0.8,  # Saturday
            6: 0.6   # Sunday
        }
        
        # Weather impact factors
        self.weather_conditions = {
            'clear': {'multiplier': 1.0, 'probability': 0.4},
            'cloudy': {'multiplier': 1.1, 'probability': 0.3},
            'rainy': {'multiplier': 0.7, 'probability': 0.15},
            'windy': {'multiplier': 0.8, 'probability': 0.1},
            'smoggy': {'multiplier': 1.8, 'probability': 0.05}
        }

    def add_seasonal_variation(self, base_value, month, day_of_year):
        """Add seasonal patterns to pollution levels"""
        # Summer months tend to have higher ozone/pollution
        summer_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Winter heating season
        winter_heating = 1 + 0.2 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        
        return base_value * summer_factor * winter_heating

    def add_weather_impact(self):
        """Randomly select weather condition and return multiplier"""
        weather_choices = list(self.weather_conditions.keys())
        probabilities = [self.weather_conditions[w]['probability'] for w in weather_choices]
        
        weather = np.random.choice(weather_choices, p=probabilities)
        return self.weather_conditions[weather]['multiplier'], weather

    def generate_correlated_gases(self, co2_level, environment, weather_multiplier):
        """Generate other gas levels that are realistically correlated with CO2"""
        env = self.environment_types[environment]
        variability = env['variability']
        
        # CO is usually well-correlated with CO2 (combustion sources)
        co_correlation = 0.7 + np.random.normal(0, 0.1)
        co = env['base_co'] * (co2_level / env['base_co2']) ** co_correlation
        co = co * weather_multiplier * (1 + np.random.normal(0, variability))
        co = max(0.5, min(co, 30.0))  # Realistic bounds
        
        # Alcohol - moderate correlation (vehicle emissions, industrial)
        alcohol_correlation = 0.5 + np.random.normal(0, 0.15)
        alcohol = env['base_alcohol'] * (co2_level / env['base_co2']) ** alcohol_correlation
        alcohol = alcohol * weather_multiplier * (1 + np.random.normal(0, variability))
        alcohol = max(0.02, min(alcohol, 1.0))
        
        # NH4 - weaker correlation (different sources)
        nh4_correlation = 0.3 + np.random.normal(0, 0.2)
        nh4 = env['base_nh4'] * (co2_level / env['base_co2']) ** nh4_correlation
        nh4 = nh4 * weather_multiplier * (1 + np.random.normal(0, variability * 1.2))
        nh4 = max(0.05, min(nh4, 1.5))
        
        # Toluen - moderate correlation (vehicle emissions, solvents)
        toluen_correlation = 0.6 + np.random.normal(0, 0.12)
        toluen = env['base_toluen'] * (co2_level / env['base_co2']) ** toluen_correlation
        toluen = toluen * weather_multiplier * (1 + np.random.normal(0, variability))
        toluen = max(0.01, min(toluen, 0.8))
        
        # Acetone - weaker correlation (industrial, solvents)
        acetone_correlation = 0.4 + np.random.normal(0, 0.18)
        acetone = env['base_acetone'] * (co2_level / env['base_co2']) ** acetone_correlation
        acetone = acetone * weather_multiplier * (1 + np.random.normal(0, variability * 1.1))
        acetone = max(0.02, min(acetone, 1.2))
        
        return {
            'CO (ppm)': round(co, 2),
            'Alkohol (ppm)': round(alcohol, 3),
            'NH4 (ppm)': round(nh4, 3),
            'Toluen (ppm)': round(toluen, 3),
            'Aseton (ppm)': round(acetone, 3)
        }

    def add_anomalies(self, data, anomaly_probability=0.05):
        """Add realistic anomalies and edge cases"""
        if np.random.random() < anomaly_probability:
            anomaly_type = np.random.choice(['sensor_drift', 'industrial_event', 'traffic_jam', 'sensor_malfunction'])
            
            if anomaly_type == 'sensor_drift':
                # Gradual sensor drift
                drift_factor = np.random.uniform(0.8, 1.3)
                for key in data:
                    if 'ppm' in key:
                        data[key] *= drift_factor
                        
            elif anomaly_type == 'industrial_event':
                # Sudden industrial emission
                spike_factor = np.random.uniform(1.5, 3.0)
                data['CO2 (ppm)'] *= spike_factor
                data['CO (ppm)'] *= spike_factor * 1.2
                data['Toluen (ppm)'] *= spike_factor * 1.5
                
            elif anomaly_type == 'traffic_jam':
                # Traffic congestion
                traffic_factor = np.random.uniform(1.3, 2.0)
                data['CO2 (ppm)'] *= traffic_factor
                data['CO (ppm)'] *= traffic_factor * 1.1
                data['Alkohol (ppm)'] *= traffic_factor * 0.8
                
            elif anomaly_type == 'sensor_malfunction':
                # Random sensor giving weird readings
                faulty_sensor = np.random.choice(['CO (ppm)', 'Alkohol (ppm)', 'NH4 (ppm)', 'Toluen (ppm)', 'Aseton (ppm)'])
                data[faulty_sensor] = np.random.uniform(0, data[faulty_sensor] * 3)
        
        return data

    def generate_dataset(self, num_records=2000, start_date='2024-01-01', environment_mix=None):
        """
        Generate a comprehensive air pollution dataset
        
        Args:
            num_records: Number of records to generate
            start_date: Starting date for the dataset
            environment_mix: Dictionary with environment type probabilities
        """
        if environment_mix is None:
            environment_mix = {
                'urban_moderate': 0.35,
                'urban_heavy': 0.20,
                'suburban': 0.25,
                'rural': 0.15,
                'industrial': 0.05
            }
        
        # Validate environment_mix
        if abs(sum(environment_mix.values()) - 1.0) > 0.01:
            raise ValueError("Environment mix probabilities must sum to 1.0")
        
        environment_types = list(environment_mix.keys())
        environment_probs = list(environment_mix.values())
        
        data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        for i in range(num_records):
            # Random time within the day
            hours_offset = np.random.uniform(0, 24 * 7)  # Week-long distribution
            timestamp = current_date + timedelta(hours=hours_offset)
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            day_of_year = timestamp.timetuple().tm_yday
            
            # Select environment type
            environment = np.random.choice(environment_types, p=environment_probs)
            env_config = self.environment_types[environment]
            
            # Get weather impact
            weather_multiplier, weather = self.add_weather_impact()
            
            # Calculate base CO2 with temporal patterns
            base_co2 = env_config['base_co2']
            base_co2 = self.add_seasonal_variation(base_co2, month, day_of_year)
            base_co2 *= self.traffic_pattern[hour]
            base_co2 *= self.day_pattern[day_of_week]
            base_co2 *= weather_multiplier
            
            # Add realistic noise
            noise = np.random.normal(0, env_config['variability'])
            co2_level = base_co2 * (1 + noise)
            co2_level = max(2.0, min(co2_level, 50.0))  # Realistic bounds
            
            # Generate correlated gas measurements
            gas_data = self.generate_correlated_gases(co2_level, environment, weather_multiplier)
            
            # Create record
            record = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'CO2 (ppm)': round(co2_level, 2),
                **gas_data
            }
            
            # Add anomalies occasionally
            record = self.add_anomalies(record)
            
            # Ensure all values are within realistic bounds
            record['CO2 (ppm)'] = max(2.0, min(record['CO2 (ppm)'], 50.0))
            record['CO (ppm)'] = max(0.5, min(record['CO (ppm)'], 30.0))
            record['Alkohol (ppm)'] = max(0.02, min(record['Alkohol (ppm)'], 1.0))
            record['NH4 (ppm)'] = max(0.05, min(record['NH4 (ppm)'], 1.5))
            record['Toluen (ppm)'] = max(0.01, min(record['Toluen (ppm)'], 0.8))
            record['Aseton (ppm)'] = max(0.02, min(record['Aseton (ppm)'], 1.2))
            
            data.append(record)
            
            # Advance date for next record
            if i % 100 == 0:  # Every 100 records, advance by a few days
                current_date += timedelta(days=np.random.uniform(0.5, 2.0))
        
        return pd.DataFrame(data)

    def save_dataset(self, df, filename='enhanced_pollution_dataset.csv'):
        """Save the generated dataset"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved as '{filename}'")
        print(f"Generated {len(df)} records")
        print("\nDataset statistics:")
        print(df.describe())
        return filename

# Generate the enhanced dataset
if __name__ == "__main__":
    generator = AirPollutionDatasetGenerator()
    
    # Generate a comprehensive dataset
    print("Generating enhanced air pollution dataset...")
    df = generator.generate_dataset(
        num_records=3000,
        start_date='2024-01-01',
        environment_mix={
            'urban_moderate': 0.35,
            'urban_heavy': 0.20,
            'suburban': 0.25,
            'rural': 0.15,
            'industrial': 0.05
        }
    )
    
    # Save the dataset
    filename = generator.save_dataset(df, 'enhanced_pollution_dataset.csv')
    
    print(f"\nEnhanced dataset created successfully!")
    print(f"Saved as: {filename}")
    print(f"Total records: {len(df)}")
    print(f"\nSample data:")
    print(df.head(3).to_string()) 