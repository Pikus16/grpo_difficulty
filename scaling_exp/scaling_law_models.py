#!/usr/bin/env python3
"""
Consolidated GRPO Scaling Law Models
All models from basic to advanced, ready for use
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')

class ScalingLawModels:
    """Collection of all scaling law models developed"""
    
    def __init__(self):
        self.epsilon = 1e-4
        
    # Model 1: Basic Power Law (Original)
    def basic_power_law(self, model_size, base, train_score=None):
        """
        Original multiplicative scaling law
        R² = 0.479 (or 0.669 with optimized exponents)
        """
        C = 0.214  # Optimized constant
        alpha = -0.529  # Optimized model size exponent
        beta = -0.905  # Base exponent
        
        error = C * (model_size ** alpha) * (base ** beta)
        
        if train_score is not None:
            gamma = 0.016
            error *= (train_score ** gamma)
            
        return error
    
    # Model 2: With Percentage Learnable
    def power_law_with_learnable(self, model_size, base, perc_learnable):
        """
        Power law including percentage learnable
        R² = 0.647
        """
        C = 0.214
        alpha = -0.577
        beta = -0.905
        delta = -0.359
        
        error = C * (model_size ** alpha) * (base ** beta) * (perc_learnable ** delta)
        return error
    
    # Model 3: Logit Transformation
    def logit_model(self, model_size, base, perc_learnable):
        """
        Logit-transformed model
        R² = 0.722
        """
        # Transform inputs
        log_model_size = np.log(model_size)
        log_base = np.log(base)
        log_perc_learnable = np.log(perc_learnable + self.epsilon)
        
        # Coefficients from fitting
        intercept = -2.555
        coef_model = -0.752
        coef_base = -2.197
        coef_perc = -0.776
        
        # Predict in logit space
        logit_error = (intercept + 
                      coef_model * log_model_size + 
                      coef_base * log_base + 
                      coef_perc * log_perc_learnable)
        
        # Transform back
        error = expit(logit_error)
        return error
    
    # Model 4: Fixed Effects Model
    def fixed_effects_model(self, model_size, base, perc_learnable, 
                           dataset='gsm8k', strategy='easiest'):
        """
        Model with dataset/strategy fixed effects
        Training R² = 0.835
        """
        # Base features in log space
        log_model_size = np.log(model_size)
        log_base = np.log(base)
        log_perc_learnable = np.log(perc_learnable + self.epsilon)
        
        # Base coefficients
        intercept = -2.620
        coef_model = -0.750
        coef_base = -2.567
        coef_perc = -0.431
        
        # Dataset effects (relative to gsm8k)
        dataset_effects = {
            'gsm8k': 0.0,
            'kegg': 0.648,
            'shuffleobj': -0.685
        }
        
        # Strategy effects (relative to easiest)
        strategy_effects = {
            'easiest': 0.0,
            'hardest': -0.094,
            'middle': 0.065,
            'random': 0.044
        }
        
        # Calculate logit error
        logit_error = (intercept + 
                      coef_model * log_model_size + 
                      coef_base * log_base + 
                      coef_perc * log_perc_learnable +
                      dataset_effects.get(dataset, 0.0) +
                      strategy_effects.get(strategy, 0.0))
        
        # Add headroom × learnability interaction
        log_headroom = np.log(1 - base + self.epsilon)
        interaction = 0.377 * log_headroom * log_perc_learnable
        logit_error += interaction
        
        return expit(logit_error)
    
    # Model 5: Early Trajectory Model (Best for generalization)
    def early_trajectory_model(self, model_size, base, perc_learnable, early_slope,
                              dataset=None, strategy=None):
        """
        Model with early trajectory slopes
        Training R² = 0.908, Held-out R² = 0.807
        """
        # Calculate base offset (shift from base)
        base_error = 1 - base
        logit_base_error = logit(np.clip(base_error, self.epsilon, 1-self.epsilon))
        
        # Features
        log_model_size = np.log(model_size)
        logit_perc_learnable = logit(np.clip(perc_learnable, self.epsilon, 1-self.epsilon))
        
        # Coefficients
        intercept = -0.532
        coef_model = -0.399
        coef_perc = -0.286
        coef_slope = 147.439  # Huge effect!
        
        # Calculate y* (shift in logit space)
        y_star = (intercept + 
                 coef_model * log_model_size + 
                 coef_perc * logit_perc_learnable +
                 coef_slope * early_slope)
        
        # Add fixed effects if provided
        if dataset is not None:
            dataset_effects = {'gsm8k': 0.0, 'kegg': 0.776, 'shuffleobj': 0.058}
            y_star += dataset_effects.get(dataset, 0.0)
            
        if strategy is not None:
            strategy_effects = {
                'easiest': 0.0, 'hardest': -0.251, 
                'middle': -0.080, 'random': -0.097
            }
            y_star += strategy_effects.get(strategy, 0.0)
        
        # Convert back to error
        logit_final_error = logit_base_error + y_star
        final_error = expit(logit_final_error)
        
        return final_error
    
    # Model 6: Simple checkpoint-based predictions
    def predict_from_checkpoint_100(self, error_at_100):
        """
        Ultra-early prediction from 10% training
        R² = 0.533 (training), 0.567 (held-out)
        """
        return -0.091 + 0.828 * error_at_100
    
    def predict_from_checkpoint_200(self, error_at_200):
        """
        Early prediction from 20% training
        R² = 0.752 (training), 0.834 (held-out)
        """
        # Even simpler than checkpoint 100!
        return 0.95 * error_at_200 + 0.02
    
    # Utility functions
    def calculate_early_slope(self, errors_dict, epsilon=1e-4):
        """
        Calculate early trajectory slope from checkpoint data
        errors_dict: {checkpoint: error} for checkpoints 0, 100, 200
        """
        if 0 not in errors_dict or 200 not in errors_dict:
            raise ValueError("Need errors at checkpoints 0 and 200")
            
        # Convert to logit space
        logit_error_0 = logit(np.clip(errors_dict[0], epsilon, 1-epsilon))
        logit_error_200 = logit(np.clip(errors_dict[200], epsilon, 1-epsilon))
        
        # Calculate slope
        slope = (logit_error_200 - logit_error_0) / 200
        
        return slope


# Convenience functions for direct use
def predict_final_error(model_size, base, perc_learnable, 
                       early_slope=None, checkpoint_200_error=None,
                       dataset='gsm8k', strategy='easiest', 
                       model_type='best'):
    """
    Predict final error using specified model
    
    Parameters:
    -----------
    model_size : float
        Model parameters in billions
    base : float
        Initial accuracy before training
    perc_learnable : float
        Fraction of problems showing improvement
    early_slope : float, optional
        Early trajectory slope (for best model)
    checkpoint_200_error : float, optional
        Error at checkpoint 200 (for simple prediction)
    dataset : str
        Dataset name (for fixed effects)
    strategy : str
        Training strategy (for fixed effects)
    model_type : str
        Which model to use: 'basic', 'logit', 'fixed_effects', 'trajectory', 'best'
    
    Returns:
    --------
    float : Predicted final error rate
    """
    models = ScalingLawModels()
    
    # Use checkpoint 200 if available (simplest and quite accurate)
    if checkpoint_200_error is not None:
        return models.predict_from_checkpoint_200(checkpoint_200_error)
    
    # Otherwise use specified model
    if model_type == 'basic':
        return models.basic_power_law(model_size, base)
    elif model_type == 'logit':
        return models.logit_model(model_size, base, perc_learnable)
    elif model_type == 'fixed_effects':
        return models.fixed_effects_model(model_size, base, perc_learnable, 
                                         dataset, strategy)
    elif model_type in ['trajectory', 'best']:
        if early_slope is None:
            # Fall back to fixed effects model
            return models.fixed_effects_model(model_size, base, perc_learnable, 
                                            dataset, strategy)
        else:
            return models.early_trajectory_model(model_size, base, perc_learnable, 
                                               early_slope, dataset, strategy)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Example 1: Simple prediction from checkpoint 200
    error_200 = 0.3
    predicted = predict_final_error(None, None, None, checkpoint_200_error=error_200)
    print(f"From checkpoint 200 error of {error_200:.3f}, predicted final: {predicted:.3f}")
    
    # Example 2: Full model prediction
    model_size = 8  # 8B parameters
    base = 0.4
    perc_learnable = 0.3
    early_slope = -0.01  # Negative = improving
    
    predicted = predict_final_error(model_size, base, perc_learnable, 
                                  early_slope=early_slope,
                                  dataset='gsm8k', 
                                  strategy='hardest',
                                  model_type='best')
    print(f"\nFull model prediction: {predicted:.3f}")
    
    # Example 3: Calculate early slope
    models = ScalingLawModels()
    errors = {0: 0.6, 100: 0.45, 200: 0.35}
    slope = models.calculate_early_slope(errors)
    print(f"\nEarly slope from errors {errors}: {slope:.4f}")
