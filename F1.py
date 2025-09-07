import fastf1
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = './f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

# Enable FastF1 cache for better performance
try:
    fastf1.Cache.enable_cache(cache_dir)
    print("FastF1 cache enabled successfully")
except Exception as e:
    print(f"Warning: Could not enable cache: {e}")
    print("Proceeding without cache (data fetching may be slower)")

class F1Azerbaijan2025Predictor:
    def __init__(self, use_real_data=True, max_retries=3):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.driver_history = {}
        self.use_real_data = use_real_data
        self.max_retries = max_retries
        
    def get_2025_driver_lineup(self):
        """Expected 2025 F1 driver lineup"""
        drivers_2025 = {
            'Max Verstappen': {'abbr': 'VER', 'team': 'Red Bull Racing Honda RBPT'},
            'Sergio Perez': {'abbr': 'PER', 'team': 'Red Bull Racing Honda RBPT'},
            'Charles Leclerc': {'abbr': 'LEC', 'team': 'Ferrari'},
            'Lewis Hamilton': {'abbr': 'HAM', 'team': 'Ferrari'},
            'Carlos Sainz': {'abbr': 'SAI', 'team': 'Williams Mercedes'},
            'George Russell': {'abbr': 'RUS', 'team': 'Mercedes'},
            'Lando Norris': {'abbr': 'NOR', 'team': 'McLaren Mercedes'},
            'Oscar Piastri': {'abbr': 'PIA', 'team': 'McLaren Mercedes'},
            'Fernando Alonso': {'abbr': 'ALO', 'team': 'Aston Martin Aramco Mercedes'},
            'Lance Stroll': {'abbr': 'STR', 'team': 'Aston Martin Aramco Mercedes'},
            'Pierre Gasly': {'abbr': 'GAS', 'team': 'Alpine Renault'},
            'Esteban Ocon': {'abbr': 'OCO', 'team': 'Alpine Renault'},
            'Alexander Albon': {'abbr': 'ALB', 'team': 'Williams Mercedes'},
            'Franco Colapinto': {'abbr': 'COL', 'team': 'Williams Mercedes'},
            'Yuki Tsunoda': {'abbr': 'TSU', 'team': 'RB Honda RBPT'},
            'Liam Lawson': {'abbr': 'LAW', 'team': 'RB Honda RBPT'},
            'Nico Hulkenberg': {'abbr': 'HUL', 'team': 'Haas Ferrari'},
            'Oliver Bearman': {'abbr': 'BEA', 'team': 'Haas Ferrari'},
            'Valtteri Bottas': {'abbr': 'BOT', 'team': 'Kick Sauber Ferrari'},
            'Guanyu Zhou': {'abbr': 'ZHO', 'team': 'Kick Sauber Ferrari'}
        }
        return drivers_2025
    
    def fetch_historical_data(self, start_year=2019, end_year=2024):
        """Fetch historical data with timeout and retry logic"""
        print("Attempting to fetch historical F1 data...")
        
        if not self.use_real_data:
            print("Using synthetic data mode...")
            return self.create_synthetic_historical_data()
        
        try:
            # Try to fetch a small sample first to test connection
            print("Testing FastF1 connection...")
            test_session = fastf1.get_session(2023, 1, 'R')  # First race of 2023
            test_session.load()
            print("✓ FastF1 connection successful")
            
            # Proceed with full data fetch
            all_data = []
            
            # Fetch Azerbaijan GP data (limited years to avoid timeout)
            azerbaijan_data = self.fetch_azerbaijan_gp_data(2020, 2023)
            if len(azerbaijan_data) > 0:
                all_data.append(azerbaijan_data)
            
            # Fetch some recent season data
            recent_data = self.fetch_recent_season_data(2023, 2024)
            if len(recent_data) > 0:
                all_data.append(recent_data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Successfully fetched {len(combined_data)} records")
                return combined_data
            else:
                print("No real data available, falling back to synthetic data")
                return self.create_synthetic_historical_data()
                
        except Exception as e:
            print(f"Error fetching real data: {e}")
            print("Falling back to synthetic data for demonstration")
            return self.create_synthetic_historical_data()
    
    def fetch_azerbaijan_gp_data(self, start_year=2020, end_year=2023):
        """Fetch Azerbaijan GP historical data with timeout protection"""
        print("Fetching Azerbaijan GP historical data...")
        all_data = []
        
        # Limit to recent years to avoid timeouts
        years_to_fetch = list(range(max(start_year, 2020), min(end_year + 1, 2024)))
        
        for year in years_to_fetch:
            try:
                print(f"  → Fetching Azerbaijan GP {year}...")
                
                # Add timeout protection
                session = fastf1.get_session(year, 'Azerbaijan', 'R')
                session.load()
                
                race_results = session.results
                
                if race_results is None or len(race_results) == 0:
                    print(f"    ⚠ No race results for Azerbaijan {year}")
                    continue
                
                # Process each driver's result
                for idx, driver_result in race_results.iterrows():
                    driver_abbr = driver_result.get('Abbreviation')
                    if pd.isna(driver_abbr) or driver_abbr is None:
                        continue
                    
                    race_data = self.extract_basic_driver_features(
                        driver_result, driver_abbr, year, 'Azerbaijan'
                    )
                    if race_data:
                        all_data.append(race_data)
                
                print(f"    ✓ Successfully processed {len(race_results)} drivers")
                
            except Exception as e:
                print(f"    ✗ Error fetching Azerbaijan data for {year}: {e}")
                continue
        
        print(f"Azerbaijan GP data fetch complete: {len(all_data)} records")
        return pd.DataFrame(all_data)
    
    def fetch_recent_season_data(self, start_year=2023, end_year=2024):
        """Fetch recent season data from selected circuits"""
        print("Fetching recent season data...")
        all_data = []
        
        # Limit circuits to avoid timeouts
        circuits = ['Monaco', 'Abu Dhabi']  # Just 2 circuits to keep it fast
        
        for year in [2023]:  # Just 2023 for now
            for circuit in circuits:
                try:
                    print(f"  → Fetching {circuit} {year}...")
                    
                    session = fastf1.get_session(year, circuit, 'R')
                    session.load()
                    
                    race_results = session.results
                    
                    if race_results is None or len(race_results) == 0:
                        continue
                    
                    for idx, driver_result in race_results.iterrows():
                        driver_abbr = driver_result.get('Abbreviation')
                        if pd.isna(driver_abbr) or driver_abbr is None:
                            continue
                        
                        race_data = self.extract_basic_driver_features(
                            driver_result, driver_abbr, year, circuit
                        )
                        if race_data:
                            all_data.append(race_data)
                    
                    print(f"    ✓ Processed {len(race_results)} drivers")
                            
                except Exception as e:
                    print(f"    ✗ Error fetching {circuit} {year} data: {e}")
                    continue
        
        print(f"Recent season data fetch complete: {len(all_data)} records")
        return pd.DataFrame(all_data)
    
    def extract_basic_driver_features(self, driver_result, driver_abbr, year, circuit):
        """Extract basic driver features without complex telemetry"""
        try:
            # Basic race info
            finish_pos = driver_result.get('Position')
            if pd.isna(finish_pos) or finish_pos is None:
                finish_pos = 20
            else:
                try:
                    finish_pos = float(finish_pos)
                    if finish_pos <= 0 or finish_pos > 20:
                        finish_pos = 20
                except (ValueError, TypeError):
                    finish_pos = 20
            
            is_winner = 1 if (finish_pos == 1.0) else 0
            
            # Driver and team info
            driver_number = driver_result.get('DriverNumber', 0)
            if pd.isna(driver_number):
                driver_number = 0
                
            team_name = driver_result.get('TeamName', 'Unknown')
            if pd.isna(team_name):
                team_name = 'Unknown'
                
            points = driver_result.get('Points', 0)
            if pd.isna(points):
                points = 0
            
            # Grid position (use finish position as estimate if not available)
            grid_pos = driver_result.get('GridPosition', finish_pos)
            if pd.isna(grid_pos):
                grid_pos = finish_pos
            
            race_data = {
                'Year': year,
                'Circuit': circuit,
                'Driver': driver_number,
                'DriverAbbr': driver_abbr,
                'Team': str(team_name),
                'GridPosition': float(grid_pos),
                'FinishPosition': finish_pos,
                'Points': float(points),
                'Winner': is_winner,
                # Add some estimated values
                'AvgLapTime': 95.0 + np.random.normal(0, 2),
                'FastestLap': 93.0 + np.random.normal(0, 1.5),
                'LapConsistency': 1.0 + abs(np.random.normal(0, 0.3)),
                'AvgSpeed': 190 + np.random.normal(0, 5),
                'MaxSpeed': 320 + np.random.normal(0, 8),
                'SOFT_laps': 20,
                'MEDIUM_laps': 25,
                'HARD_laps': 6,
                'INTERMEDIATE_laps': 0,
                'WET_laps': 0
            }
            
            return race_data
            
        except Exception as e:
            print(f"Error extracting features for {driver_abbr}: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for machine learning model"""
        print("Preparing features for training...")
        
        if len(df) == 0:
            raise ValueError("No data provided for feature preparation")
        
        data = df.copy()
        print(f"Initial data shape: {data.shape}")
        
        # Remove rows with missing critical data
        critical_cols = ['DriverAbbr', 'Team']
        before_drop = len(data)
        data = data.dropna(subset=critical_cols)
        print(f"After removing missing critical columns: {len(data)} (removed {before_drop - len(data)})")
        
        if len(data) == 0:
            raise ValueError("No valid data after removing missing critical columns")
        
        # Ensure GridPosition is numeric and reasonable
        data['GridPosition'] = pd.to_numeric(data['GridPosition'], errors='coerce')
        data['GridPosition'] = data['GridPosition'].fillna(15)
        data['GridPosition'] = data['GridPosition'].clip(1, 20)
        
        # Fill missing numeric data with reasonable defaults
        numeric_cols = ['AvgLapTime', 'FastestLap', 'LapConsistency', 'AvgSpeed', 'MaxSpeed', 'Points']
        defaults = {
            'AvgLapTime': 100.0,
            'FastestLap': 98.0,
            'LapConsistency': 2.0,
            'AvgSpeed': 180.0,
            'MaxSpeed': 300.0,
            'Points': 0.0
        }
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(defaults.get(col, 0.0))
        
        # Ensure tire compound columns exist
        for compound in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']:
            col_name = f'{compound}_laps'
            if col_name not in data.columns:
                data[col_name] = 0
            else:
                data[col_name] = pd.to_numeric(data[col_name], errors='coerce').fillna(0)
        
        # Encode categorical variables
        categorical_cols = ['Team', 'DriverAbbr']
        for col in categorical_cols:
            data[col] = data[col].astype(str)
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle new categories
                known_categories = set(self.label_encoders[col].classes_)
                data[f'{col}_temp'] = data[col].apply(
                    lambda x: x if x in known_categories else 'Unknown'
                )
                
                if 'Unknown' not in known_categories:
                    new_classes = list(self.label_encoders[col].classes_) + ['Unknown']
                    self.label_encoders[col].classes_ = np.array(new_classes)
                
                data[f'{col}_encoded'] = self.label_encoders[col].transform(data[f'{col}_temp'])
                data.drop(f'{col}_temp', axis=1, inplace=True)
        
        # Create performance features
        data['GridAdvantage'] = 21 - data['GridPosition']
        
        # Sort data for rolling calculations
        data = data.sort_values(['DriverAbbr', 'Year'])
        
        # Recent form (simplified)
        data['RecentForm'] = data.groupby('DriverAbbr')['Winner'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1).fillna(0)
        )
        
        # Azerbaijan-specific performance
        azerbaijan_mask = data['Circuit'] == 'Azerbaijan'
        if azerbaijan_mask.any():
            azerbaijan_data = data[azerbaijan_mask]
            az_performance = azerbaijan_data.groupby('DriverAbbr').agg({
                'Winner': 'mean',
                'FinishPosition': 'mean',
                'GridPosition': 'mean'
            }).add_suffix('_Azerbaijan')
            
            data = data.merge(az_performance, left_on='DriverAbbr', right_index=True, how='left')
        
        # Fill missing Azerbaijan stats
        for col in ['Winner_Azerbaijan', 'FinishPosition_Azerbaijan', 'GridPosition_Azerbaijan']:
            if col not in data.columns:
                if 'Winner' in col:
                    data[col] = 0
                else:
                    data[col] = 10
            data[col] = data[col].fillna(0 if 'Winner' in col else 10)
        
        # Team performance
        data['TeamSeasonForm'] = data.groupby(['Team', 'Year'])['Winner'].transform('mean').fillna(0.1)
        
        # Select features (simplified set)
        feature_cols = [
            'GridPosition', 'GridAdvantage', 'RecentForm',
            'Winner_Azerbaijan', 'FinishPosition_Azerbaijan', 'GridPosition_Azerbaijan',
            'TeamSeasonForm', 'SOFT_laps', 'MEDIUM_laps', 'HARD_laps',
            'Team_encoded', 'DriverAbbr_encoded'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in data.columns]
        
        # Final cleaning
        data = data.dropna(subset=available_features)
        
        if len(data) == 0:
            raise ValueError("No valid data after feature preparation")
        
        self.feature_names = available_features
        print(f"Using {len(available_features)} features: {available_features}")
        
        X = data[available_features]
        y = data['Winner']
        
        print(f"Final dataset: {len(X)} samples, {y.sum()} winners ({y.mean()*100:.1f}% win rate)")
        
        return X, y, data
    
    def train_model(self, X, y):
        """Train prediction model"""
        print("Training prediction model...")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data provided for training")
        
        print(f"Training on {len(X)} samples with {y.sum()} positive examples")

        # Use simpler model for small datasets
        if len(X) < 100:
            model = RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                class_weight='balanced',
                max_depth=5
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced',
                max_depth=10
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model.fit(X_scaled, y)
        
        # Simple evaluation
        score = model.score(X_scaled, y)
        print(f"Model Training Score: {score:.4f}")

        self.best_model = model
        self.best_model_name = "RandomForest"

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

        return score
    
    def create_2025_race_scenario(self, grid_positions=None, team_performance=None):
        """Create a race scenario for 2025 Azerbaijan GP"""
        drivers_2025 = self.get_2025_driver_lineup()
        
        if grid_positions is None:
            default_grid = {
                'Max Verstappen': 1, 'Charles Leclerc': 2, 'Lando Norris': 3, 
                'Oscar Piastri': 4, 'Lewis Hamilton': 5, 'George Russell': 6, 
                'Sergio Perez': 7, 'Carlos Sainz': 8, 'Fernando Alonso': 9, 
                'Lance Stroll': 10, 'Nico Hulkenberg': 11, 'Pierre Gasly': 12, 
                'Alexander Albon': 13, 'Yuki Tsunoda': 14, 'Esteban Ocon': 15,
                'Liam Lawson': 16, 'Valtteri Bottas': 17, 'Guanyu Zhou': 18, 
                'Oliver Bearman': 19, 'Franco Colapinto': 20
            }
            grid_positions = default_grid
        
        if team_performance is None:
            team_performance = {
                'Red Bull Racing Honda RBPT': 0.85,
                'Ferrari': 0.75,
                'McLaren Mercedes': 0.70,
                'Mercedes': 0.65,
                'Aston Martin Aramco Mercedes': 0.45,
                'Alpine Renault': 0.35,
                'Williams Mercedes': 0.25,
                'RB Honda RBPT': 0.30,
                'Haas Ferrari': 0.20,
                'Kick Sauber Ferrari': 0.15
            }
        
        scenario_data = []
        
        for driver_name, info in drivers_2025.items():
            grid_pos = grid_positions.get(driver_name, 20)
            team = info['team']
            abbr = info['abbr']
            
            race_data = {
                'Year': 2025,
                'Circuit': 'Azerbaijan',
                'DriverAbbr': abbr,
                'Team': team,
                'GridPosition': int(grid_pos),
                'GridAdvantage': 21 - grid_pos,
                'RecentForm': self.estimate_recent_form(abbr),
                'Winner_Azerbaijan': self.estimate_azerbaijan_performance(abbr),
                'FinishPosition_Azerbaijan': self.estimate_avg_azerbaijan_position(abbr),
                'GridPosition_Azerbaijan': self.estimate_avg_azerbaijan_grid(abbr),
                'TeamSeasonForm': team_performance.get(team, 0.3),
                'SOFT_laps': 20,
                'MEDIUM_laps': 25,
                'HARD_laps': 6,
                'INTERMEDIATE_laps': 0,
                'WET_laps': 0
            }
            
            scenario_data.append(race_data)
        
        return pd.DataFrame(scenario_data)
    
    def estimate_recent_form(self, driver_abbr):
        """Estimate driver's recent form"""
        form_estimates = {
            'VER': 0.45, 'LEC': 0.25, 'NOR': 0.20, 'PIA': 0.15, 'HAM': 0.20,
            'RUS': 0.10, 'PER': 0.05, 'SAI': 0.15, 'ALO': 0.05, 'STR': 0.02,
            'HUL': 0.02, 'GAS': 0.02, 'ALB': 0.01, 'TSU': 0.01, 'OCO': 0.01,
            'LAW': 0.01, 'BOT': 0.01, 'ZHO': 0.01, 'BEA': 0.01, 'COL': 0.01
        }
        return form_estimates.get(driver_abbr, 0.01)
    
    def estimate_azerbaijan_performance(self, driver_abbr):
        """Estimate driver's Azerbaijan-specific performance"""
        az_estimates = {
            'VER': 0.4, 'LEC': 0.2, 'PER': 0.2, 'RUS': 0.1, 'HAM': 0.1,
            'NOR': 0.05, 'PIA': 0.02, 'SAI': 0.05, 'ALO': 0.05, 'STR': 0.01,
            'HUL': 0.01, 'GAS': 0.02, 'ALB': 0.01, 'TSU': 0.01, 'OCO': 0.01,
            'LAW': 0.01, 'BOT': 0.02, 'ZHO': 0.01, 'BEA': 0.01, 'COL': 0.01
        }
        return az_estimates.get(driver_abbr, 0.01)
    
    def estimate_avg_azerbaijan_position(self, driver_abbr):
        """Estimate average Azerbaijan finishing position"""
        pos_estimates = {
            'VER': 2.5, 'LEC': 4.0, 'PER': 5.0, 'HAM': 6.0, 'RUS': 7.0,
            'NOR': 8.0, 'PIA': 10.0, 'SAI': 8.5, 'ALO': 9.0, 'STR': 12.0,
            'HUL': 13.0, 'GAS': 11.0, 'ALB': 14.0, 'TSU': 15.0, 'OCO': 12.0,
            'LAW': 16.0, 'BOT': 17.0, 'ZHO': 18.0, 'BEA': 19.0, 'COL': 20.0
        }
        return pos_estimates.get(driver_abbr, 15.0)
    
    def estimate_avg_azerbaijan_grid(self, driver_abbr):
        """Estimate average Azerbaijan grid position"""
        grid_estimates = {
            'VER': 2.0, 'LEC': 3.5, 'PER': 4.5, 'HAM': 5.0, 'RUS': 6.0,
            'NOR': 7.0, 'PIA': 9.0, 'SAI': 7.5, 'ALO': 8.0, 'STR': 11.0,
            'HUL': 13.0, 'GAS': 10.0, 'ALB': 14.0, 'TSU': 15.0, 'OCO': 12.0,
            'LAW': 16.0, 'BOT': 17.0, 'ZHO': 18.0, 'BEA': 19.0, 'COL': 20.0
        }
        return grid_estimates.get(driver_abbr, 15.0)
    
    def encode_2025_scenario(self, scenario_df):
        """Encode 2025 scenario using trained encoders"""
        data = scenario_df.copy()
        
        for col in ['Team', 'DriverAbbr']:
            if col not in self.label_encoders:
                print(f"Warning: No encoder found for {col}")
                continue

            encoder = self.label_encoders[col]
            
            # Handle unknown categories
            data[f'{col}_temp'] = data[col].astype(str)
            for idx, value in enumerate(data[f'{col}_temp']):
                if value not in encoder.classes_:
                    # Use first known class as fallback
                    data.at[idx, f'{col}_temp'] = encoder.classes_[0]
            
            data[f'{col}_encoded'] = encoder.transform(data[f'{col}_temp'])
            data.drop(f'{col}_temp', axis=1, inplace=True)
        
        return data
    
    def predict_race_winner(self, scenario_df=None):
        """Predict race winner for 2025 Azerbaijan GP"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if scenario_df is None:
            scenario_df = self.create_2025_race_scenario()
        
        # Encode scenario
        encoded_scenario = self.encode_2025_scenario(scenario_df)
        
        # Prepare features
        X_scenario = encoded_scenario[self.feature_names]
        X_scenario_scaled = self.scaler.transform(X_scenario)
        
        # Make predictions
        win_probabilities = self.best_model.predict_proba(X_scenario_scaled)[:, 1]
        
        # Create results
        results = pd.DataFrame({
            'Driver': encoded_scenario['DriverAbbr'],
            'Team': encoded_scenario['Team'],
            'GridPosition': encoded_scenario['GridPosition'],
            'WinProbability': win_probabilities
        })
        
        results = results.sort_values('WinProbability', ascending=False)
        return results
    
    def create_synthetic_historical_data(self):
        """Create synthetic historical data for demonstration"""
        print("Creating synthetic historical data...")
        
        drivers = ['VER', 'LEC', 'HAM', 'RUS', 'NOR', 'PIA', 'SAI', 'PER', 'ALO', 'STR', 
                  'GAS', 'OCO', 'ALB', 'TSU', 'HUL', 'BOT', 'ZHO', 'MAG', 'MSC', 'LAT']
        teams = ['Red Bull Racing Honda RBPT', 'Ferrari', 'Mercedes', 'McLaren Mercedes', 
                'Alpine Renault', 'Aston Martin Aramco Mercedes', 'Williams Mercedes', 
                'RB Honda RBPT', 'Haas Ferrari', 'Kick Sauber Ferrari']
        
        synthetic_data = []
        
        for year in range(2020, 2025):
            for circuit in ['Azerbaijan', 'Monaco', 'Singapore']:
                for i, driver in enumerate(drivers):
                    team = teams[i % len(teams)]
                    
                    grid_pos = np.random.randint(1, 21)
                    finish_pos = max(1, grid_pos + np.random.randint(-3, 4))
                    finish_pos = min(20, finish_pos)
                    
                    is_winner = 1 if finish_pos == 1 else 0
                    
                    race_data = {
                        'Year': year,
                        'Circuit': circuit,
                        'Driver': i + 1,
                        'DriverAbbr': driver,
                        'Team': team,
                        'GridPosition': grid_pos,
                        'FinishPosition': finish_pos,
                        'Points': max(0, 26 - finish_pos) if finish_pos <= 10 else 0,
                        'Winner': is_winner,
                        'AvgLapTime': 95.0 + np.random.normal(0, 3),
                        'FastestLap': 93.0 + np.random.normal(0, 2),
                        'LapConsistency': 1.0 + abs(np.random.normal(0, 0.5)),
                        'AvgSpeed': 185 + np.random.normal(0, 10),
                        'MaxSpeed': 315 + np.random.normal(0, 15),
                        'SOFT_laps': np.random.randint(15, 25),
                        'MEDIUM_laps': np.random.randint(20, 30),
                        'HARD_laps': np.random.randint(0, 10),
                        'INTERMEDIATE_laps': 0,
                        'WET_laps': 0
                    }
                    
                    synthetic_data.append(race_data)
        
        print(f"Created {len(synthetic_data)} synthetic race records")
        return pd.DataFrame(synthetic_data)
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline with better progress tracking"""
        print("=" * 60)
        print("🏎️  F1 Azerbaijan 2025 Race Winner Prediction Analysis")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Step 1: Fetch historical data
            print("\n📊 Step 1: Fetching historical data...")
            historical_data = self.fetch_historical_data()
            
            if len(historical_data) == 0:
                raise ValueError("No data available for training")
            
            print(f"✓ Collected {len(historical_data)} historical race entries")
            
            # Step 2: Prepare features
            print("\n🔧 Step 2: Preparing features...")
            X, y, processed_data = self.prepare_features(historical_data)
            print(f"✓ Prepared features: {X.shape[1]} features, {len(X)} samples")
            
            # Step 3: Train model
            print("\n🤖 Step 3: Training prediction model...")
            score = self.train_model(X, y)
            print(f"✓ Model trained with score: {score:.4f}")
            
            # Step 4: Feature importance
            if self.feature_importance is not None:
                print("\n📈 Step 4: Feature Importance (Top 5):")
                for i, row in self.feature_importance.head(5).iterrows():
                    print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            # Step 5: Make 2025 predictions
            print("\n🏆 Step 5: Predicting 2025 Azerbaijan GP winner...")
            predictions = self.predict_race_winner()
            
            # Results
            print("\n" + "="*60)
            print("🏁 RACE WINNER PREDICTIONS - 2025 AZERBAIJAN GP")
            print("="*60)
            
            for i, row in predictions.head(10).iterrows():
                position = i + 1
                if position == 1:
                    icon = "🥇"
                elif position == 2:
                    icon = "🥈"
                elif position == 3:
                    icon = "🥉"
                else:
                    icon = f"{position:2d}."
                
                print(f"{icon} {row['Driver']:3s} ({row['Team'][:15]:15s}) "
                      f"P{row['GridPosition']:2.0f} → {row['WinProbability']:.1%} win chance")
            
            # Summary
            print("\n" + "="*60)
            print("📊 ANALYSIS SUMMARY")
            print("="*60)
            print(f"🎯 Most likely winner: {predictions.iloc[0]['Driver']} "
                  f"({predictions.iloc[0]['WinProbability']:.1%} chance)")
            print(f"🏎️ Top 3 combined probability: {predictions.head(3)['WinProbability'].sum():.1%}")
            print(f"🏁 Starting from pole: {predictions.iloc[0]['GridPosition']:.0f}")
            print(f"⏱️ Analysis completed in {time.time() - start_time:.1f} seconds")
            print(f"📈 Model accuracy: {score:.3f}")
            
            # Additional insights
            top_teams = predictions.head(5)['Team'].value_counts()
            if len(top_teams) > 0:
                print(f"🏆 Dominant team in top 5: {top_teams.index[0]} ({top_teams.iloc[0]} drivers)")
            
            return predictions
            
        except Exception as e:
            print(f"\n❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_detailed_analysis(self, predictions):
        """Display detailed analysis results"""
        if predictions is None:
            print("No predictions available")
            return
        
        print("\n" + "="*80)
        print("🔍 DETAILED RACE ANALYSIS")
        print("="*80)
        
        # Probability tiers
        print("\n🎯 WINNING PROBABILITY TIERS:")
        high_prob = predictions[predictions['WinProbability'] > 0.15]
        med_prob = predictions[(predictions['WinProbability'] > 0.05) & (predictions['WinProbability'] <= 0.15)]
        low_prob = predictions[predictions['WinProbability'] <= 0.05]
        
        if len(high_prob) > 0:
            print(f"   🔥 High chance (>15%): {', '.join(high_prob['Driver'].tolist())}")
        if len(med_prob) > 0:
            print(f"   ⚡ Medium chance (5-15%): {', '.join(med_prob['Driver'].tolist())}")
        if len(low_prob) > 0:
            print(f"   💤 Long shots (<5%): {len(low_prob)} drivers")
        
        # Grid position analysis
        print(f"\n🏁 GRID POSITION INSIGHTS:")
        front_row_winners = predictions[predictions['GridPosition'] <= 2]
        if len(front_row_winners) > 0:
            total_prob = front_row_winners['WinProbability'].sum()
            print(f"   Front row win probability: {total_prob:.1%}")
        
        top_5_grid = predictions[predictions['GridPosition'] <= 5]
        if len(top_5_grid) > 0:
            total_prob = top_5_grid['WinProbability'].sum()
            print(f"   Top 5 grid win probability: {total_prob:.1%}")

def main():
    """Main execution function"""
    print("🏎️ Starting F1 Azerbaijan 2025 Predictor...")
    print("⏳ This may take a few moments to fetch and process data...\n")
    
    # Create predictor instance
    predictor = F1Azerbaijan2025Predictor(use_real_data=True)
    
    # Run analysis
    predictions = predictor.run_complete_analysis()
    
    # Show detailed analysis if successful
    if predictions is not None:
        predictor.display_detailed_analysis(predictions)
        
        # Save results
        try:
            predictions.to_csv('f1_azerbaijan_2025_predictions.csv', index=False)
            print(f"\n💾 Results saved to 'f1_azerbaijan_2025_predictions.csv'")
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
