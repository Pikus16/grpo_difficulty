#!/usr/bin/env python3
"""
Advanced scaling models: fixed effects and early trajectory models
Consolidated from expert advice implementations
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')


class FixedEffectsModel:
    """Scaling law with dataset/strategy fixed effects"""
    
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.epsilon = 1e-4
        self.model = None
        self.dataset_effects = {}
        self.strategy_effects = {}
        self.coefficients = {}
        
    def fit(self, df):
        """Fit the fixed effects model"""
        # Prepare data
        final_df = df[df['checkpoint'] == 1000].copy()
        final_df['error'] = 1 - final_df['final_acc']
        final_df['error_clipped'] = np.clip(final_df['error'], self.epsilon, 1-self.epsilon)
        final_df['logit_error'] = logit(final_df['error_clipped'])
        
        # Base features
        X_base = np.log(final_df[['model_size', 'base', 'perc_learnable']].values + 1e-8)
        
        # Create dummy variables
        dataset_dummies = pd.get_dummies(final_df['dataset'], prefix='dataset', drop_first=True)
        strategy_dummies = pd.get_dummies(final_df['strategy'], prefix='strategy', drop_first=True)
        
        # Headroom × learnability interaction
        final_df['log_headroom'] = np.log(1 - final_df['base'] + 1e-8)
        final_df['log_perc_learnable'] = np.log(final_df['perc_learnable'] + 1e-8)
        final_df['interaction'] = final_df['log_headroom'] * final_df['log_perc_learnable']
        
        # Combine features
        X = np.hstack([
            X_base,
            dataset_dummies.values,
            strategy_dummies.values,
            final_df['interaction'].values.reshape(-1, 1)
        ])
        
        # Fit model
        y = final_df['logit_error'].values
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        
        # Extract coefficients
        self.coefficients = {
            'intercept': self.model.intercept_,
            'model_size': self.model.coef_[0],
            'base': self.model.coef_[1],
            'perc_learnable': self.model.coef_[2],
            'interaction': self.model.coef_[-1]
        }
        
        # Extract dataset effects
        ref_dataset = 'gsm8k'  # Reference category
        self.dataset_effects = {ref_dataset: 0.0}
        for i, col in enumerate(dataset_dummies.columns):
            dataset = col.replace('dataset_', '')
            self.dataset_effects[dataset] = self.model.coef_[3 + i]
        
        # Extract strategy effects
        ref_strategy = 'easiest'  # Reference category
        self.strategy_effects = {ref_strategy: 0.0}
        for i, col in enumerate(strategy_dummies.columns):
            strategy = col.replace('strategy_', '')
            self.strategy_effects[strategy] = self.model.coef_[3 + len(dataset_dummies.columns) + i]
        
        # Calculate R²
        pred = expit(self.model.predict(X))
        self.train_r2 = r2_score(final_df['error'].values, pred)
        
        return self
    
    def predict(self, model_size, base, perc_learnable, dataset='gsm8k', strategy='easiest'):
        """Predict error using the fixed effects model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Base features
        log_features = np.log(np.array([model_size, base, perc_learnable]) + 1e-8)
        
        # Calculate prediction
        logit_pred = (self.coefficients['intercept'] +
                     self.coefficients['model_size'] * log_features[0] +
                     self.coefficients['base'] * log_features[1] +
                     self.coefficients['perc_learnable'] * log_features[2])
        
        # Add fixed effects
        logit_pred += self.dataset_effects.get(dataset, 0.0)
        logit_pred += self.strategy_effects.get(strategy, 0.0)
        
        # Add interaction
        log_headroom = np.log(1 - base + 1e-8)
        log_perc = np.log(perc_learnable + 1e-8)
        logit_pred += self.coefficients['interaction'] * log_headroom * log_perc
        
        return expit(logit_pred)


class EarlyTrajectoryModel:
    """Model using early trajectory slopes (best for generalization)"""
    
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.epsilon = 1e-4
        self.model = None
        self.coefficients = {}
        self.dataset_effects = {}
        self.strategy_effects = {}
        
    def calculate_early_slope(self, df, dataset, strategy, model_name):
        """Calculate early trajectory slope for a specific run"""
        # Get run data
        mask = (df['dataset'] == dataset) & \
               (df['strategy'] == strategy) & \
               (df['model_name'] == model_name)
        run_data = df[mask]
        
        # Get checkpoints 0 (base) and 200
        base = run_data.iloc[0]['base']
        cp_200 = run_data[run_data['checkpoint'] == 200]
        
        if len(cp_200) == 0:
            return None
            
        # Calculate slope in logit space
        error_0 = 1 - base
        error_200 = 1 - cp_200.iloc[0]['accuracy']
        
        logit_0 = logit(np.clip(error_0, self.epsilon, 1-self.epsilon))
        logit_200 = logit(np.clip(error_200, self.epsilon, 1-self.epsilon))
        
        return (logit_200 - logit_0) / 200
    
    def fit(self, train_df, held_out_df=None):
        """Fit the early trajectory model"""
        # Extract features for training data
        train_features = self._extract_features(train_df)
        
        if len(train_features) == 0:
            raise ValueError("No valid training data with early slopes")
        
        # Prepare features
        train_features['y_star'] = (logit(np.clip(train_features['final_error'], self.epsilon, 1-self.epsilon)) -
                                   logit(np.clip(train_features['base_error'], self.epsilon, 1-self.epsilon)))
        
        # Create feature matrix
        X_train = train_features[['log_model_size', 'logit_perc_learnable', 'early_slope']].values
        y_train = train_features['y_star'].values
        
        # Add dataset/strategy dummies
        dataset_dummies = pd.get_dummies(train_features['dataset'], prefix='dataset', drop_first=True)
        strategy_dummies = pd.get_dummies(train_features['strategy'], prefix='strategy', drop_first=True)
        
        X_train = np.hstack([X_train, dataset_dummies.values, strategy_dummies.values])
        
        # Fit model
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)
        
        # Extract coefficients
        self.coefficients = {
            'intercept': self.model.intercept_,
            'model_size': self.model.coef_[0],
            'perc_learnable': self.model.coef_[1],
            'early_slope': self.model.coef_[2]
        }
        
        # Extract fixed effects
        self._extract_fixed_effects(dataset_dummies.columns, strategy_dummies.columns)
        
        # Calculate training R²
        train_pred_star = self.model.predict(X_train)
        train_pred_error = expit(logit(np.clip(train_features['base_error'], self.epsilon, 1-self.epsilon)) + 
                                train_pred_star)
        self.train_r2 = r2_score(train_features['final_error'], train_pred_error)
        
        # Calculate held-out R² if provided
        if held_out_df is not None:
            held_features = self._extract_features(held_out_df)
            if len(held_features) > 0:
                self.held_out_r2 = self._evaluate_held_out(held_features, dataset_dummies.columns, 
                                                          strategy_dummies.columns)
            else:
                self.held_out_r2 = None
        
        return self
    
    def _extract_features(self, df):
        """Extract features from dataframe"""
        features = []
        
        # Get unique runs
        final_df = df[df['checkpoint'] == 1000]
        
        for _, row in final_df.iterrows():
            # Calculate early slope
            slope = self.calculate_early_slope(df, row['dataset'], 
                                             row['strategy'], row['model_name'])
            
            if slope is not None:
                features.append({
                    'dataset': row['dataset'],
                    'strategy': row['strategy'],
                    'model_name': row['model_name'],
                    'model_size': row['model_size'],
                    'base': row['base'],
                    'perc_learnable': row['perc_learnable'],
                    'final_error': 1 - row['final_acc'],
                    'base_error': 1 - row['base'],
                    'early_slope': slope,
                    'log_model_size': np.log(row['model_size']),
                    'logit_perc_learnable': logit(np.clip(row['perc_learnable'], 
                                                         self.epsilon, 1-self.epsilon))
                })
        
        return pd.DataFrame(features)
    
    def _extract_fixed_effects(self, dataset_cols, strategy_cols):
        """Extract fixed effects from model coefficients"""
        # Dataset effects
        self.dataset_effects = {'gsm8k': 0.0}  # Reference
        for i, col in enumerate(dataset_cols):
            dataset = col.replace('dataset_', '')
            self.dataset_effects[dataset] = self.model.coef_[3 + i]
        
        # Strategy effects
        self.strategy_effects = {'easiest': 0.0}  # Reference
        for i, col in enumerate(strategy_cols):
            strategy = col.replace('strategy_', '')
            self.strategy_effects[strategy] = self.model.coef_[3 + len(dataset_cols) + i]
    
    def _evaluate_held_out(self, held_features, dataset_cols, strategy_cols):
        """Evaluate on held-out data"""
        # Prepare features
        held_features['y_star'] = (logit(np.clip(held_features['final_error'], self.epsilon, 1-self.epsilon)) -
                                  logit(np.clip(held_features['base_error'], self.epsilon, 1-self.epsilon)))
        
        X_held = held_features[['log_model_size', 'logit_perc_learnable', 'early_slope']].values
        
        # Add dummies (handling missing categories)
        held_dataset_dummies = pd.get_dummies(held_features['dataset'], prefix='dataset', drop_first=True)
        held_strategy_dummies = pd.get_dummies(held_features['strategy'], prefix='strategy', drop_first=True)
        
        # Align columns
        for col in dataset_cols:
            if col not in held_dataset_dummies.columns:
                held_dataset_dummies[col] = 0
        for col in strategy_cols:
            if col not in held_strategy_dummies.columns:
                held_strategy_dummies[col] = 0
        
        X_held = np.hstack([
            X_held,
            held_dataset_dummies[dataset_cols].values,
            held_strategy_dummies[strategy_cols].values
        ])
        
        # Predict
        held_pred_star = self.model.predict(X_held)
        held_pred_error = expit(logit(np.clip(held_features['base_error'], self.epsilon, 1-self.epsilon)) + 
                               held_pred_star)
        
        return r2_score(held_features['final_error'], held_pred_error)
    
    def predict(self, model_size, base, perc_learnable, early_slope, 
                dataset=None, strategy=None):
        """Predict using early trajectory model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Base features
        features = [
            np.log(model_size),
            logit(np.clip(perc_learnable, self.epsilon, 1-self.epsilon)),
            early_slope
        ]
        
        # Calculate y*
        base_error = 1 - base
        logit_base = logit(np.clip(base_error, self.epsilon, 1-self.epsilon))
        
        y_star = (self.coefficients['intercept'] +
                 self.coefficients['model_size'] * features[0] +
                 self.coefficients['perc_learnable'] * features[1] +
                 self.coefficients['early_slope'] * features[2])
        
        # Add fixed effects if available
        if dataset is not None:
            y_star += self.dataset_effects.get(dataset, 0.0)
        if strategy is not None:
            y_star += self.strategy_effects.get(strategy, 0.0)
        
        # Convert back to error
        logit_final = logit_base + y_star
        return expit(logit_final)


# Example usage
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('../scaling_analysis_results.csv')
    held_out_df = pd.read_csv('../held_out_scaling_numbers.csv')
    
    # Test fixed effects model
    print("Testing Fixed Effects Model...")
    fixed_model = FixedEffectsModel()
    fixed_model.fit(train_df)
    print(f"Training R² = {fixed_model.train_r2:.4f}")
    print(f"Dataset effects: {fixed_model.dataset_effects}")
    print(f"Strategy effects: {fixed_model.strategy_effects}")
    
    # Test early trajectory model
    print("\nTesting Early Trajectory Model...")
    traj_model = EarlyTrajectoryModel()
    traj_model.fit(train_df, held_out_df)
    print(f"Training R² = {traj_model.train_r2:.4f}")
    if hasattr(traj_model, 'held_out_r2') and traj_model.held_out_r2 is not None:
        print(f"Held-out R² = {traj_model.held_out_r2:.4f}")
    print(f"Early slope coefficient: {traj_model.coefficients['early_slope']:.1f}")
