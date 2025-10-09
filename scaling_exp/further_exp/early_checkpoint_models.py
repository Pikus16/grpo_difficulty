#!/usr/bin/env python3
"""
Early checkpoint prediction models and analysis
Consolidated from multiple checkpoint analysis scripts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.special import logit, expit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class EarlyCheckpointPredictor:
    """Models for predicting final performance from early checkpoints"""
    
    def __init__(self):
        self.epsilon = 1e-4
        self.models = {}
        
    def fit_checkpoint_model(self, df, checkpoint):
        """Fit a simple linear model for a specific checkpoint"""
        # Get checkpoint and final data
        cp_data = df[df['checkpoint'] == checkpoint].copy()
        final_data = df[df['checkpoint'] == 1000].copy()
        
        # Merge on run identifiers
        merged = cp_data.merge(
            final_data[['dataset', 'strategy', 'model_name', 'final_acc']], 
            on=['dataset', 'strategy', 'model_name']
        )
        
        if len(merged) == 0:
            return None
            
        # Calculate errors
        merged[f'error_{checkpoint}'] = 1 - merged['accuracy']
        merged['final_error'] = 1 - merged['final_acc']
        
        # Fit simple linear model
        lr = LinearRegression()
        X = merged[f'error_{checkpoint}'].values.reshape(-1, 1)
        y = merged['final_error'].values
        lr.fit(X, y)
        
        # Store model and stats
        self.models[checkpoint] = {
            'model': lr,
            'r2': lr.score(X, y),
            'n_samples': len(merged),
            'equation': f"final_error = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} × error@{checkpoint}"
        }
        
        return lr
    
    def predict_from_checkpoint(self, error_at_checkpoint, checkpoint):
        """Predict final error from checkpoint error"""
        if checkpoint not in self.models:
            raise ValueError(f"No model fitted for checkpoint {checkpoint}")
            
        model = self.models[checkpoint]['model']
        return model.predict([[error_at_checkpoint]])[0]
    
    def get_checkpoint_r2(self, checkpoint):
        """Get R² for a checkpoint model"""
        if checkpoint not in self.models:
            return None
        return self.models[checkpoint]['r2']
    
    def analyze_all_checkpoints(self, df, checkpoints=[100, 200, 300, 500]):
        """Analyze predictive power across multiple checkpoints"""
        results = []
        
        for cp in checkpoints:
            model = self.fit_checkpoint_model(df, cp)
            if model is not None:
                stats = self.models[cp]
                results.append({
                    'checkpoint': cp,
                    'percent_training': cp / 10,
                    'r2': stats['r2'],
                    'n_samples': stats['n_samples'],
                    'slope': model.coef_[0],
                    'intercept': model.intercept_,
                    'equation': stats['equation']
                })
        
        return pd.DataFrame(results)


def analyze_trajectory_features(df):
    """Extract and analyze trajectory-based features"""
    epsilon = 1e-4
    features_list = []
    
    # Group by run
    for (dataset, strategy, model_name), group in df.groupby(['dataset', 'strategy', 'model_name']):
        # Check if we have required checkpoints
        checkpoints = set(group['checkpoint'].values)
        if not all(cp in checkpoints for cp in [100, 200, 1000]):
            continue
            
        # Get checkpoint data
        base = group.iloc[0]['base']
        cp_100 = group[group['checkpoint'] == 100].iloc[0]
        cp_200 = group[group['checkpoint'] == 200].iloc[0]
        cp_final = group[group['checkpoint'] == 1000].iloc[0]
        
        # Calculate errors
        error_0 = 1 - base
        error_100 = 1 - cp_100['accuracy']
        error_200 = 1 - cp_200['accuracy']
        error_final = 1 - cp_final['final_acc']
        
        # Logit space features
        logit_error_0 = logit(np.clip(error_0, epsilon, 1-epsilon))
        logit_error_100 = logit(np.clip(error_100, epsilon, 1-epsilon))
        logit_error_200 = logit(np.clip(error_200, epsilon, 1-epsilon))
        
        # Calculate trajectory features
        features = {
            'dataset': dataset,
            'strategy': strategy,
            'model_name': model_name,
            'model_size': cp_100['model_size'],
            
            # Raw errors
            'error_100': error_100,
            'error_200': error_200,
            'final_error': error_final,
            
            # Improvements
            'improve_0_100': base - error_100,
            'improve_100_200': error_100 - error_200,
            'improve_0_200': base - error_200,
            'improve_200_1000': error_200 - error_final,
            
            # Slopes
            'slope_0_100': (base - error_100) / 100,
            'slope_100_200': (error_100 - error_200) / 100,
            'slope_0_200': (base - error_200) / 200,
            
            # Logit slopes
            'logit_slope_0_100': (logit_error_100 - logit_error_0) / 100,
            'logit_slope_100_200': (logit_error_200 - logit_error_100) / 100,
            'logit_slope_0_200': (logit_error_200 - logit_error_0) / 200,
            
            # Curvature
            'acceleration': ((error_200 - error_100) - (error_100 - error_0)) / 100,
            
            # Other features
            'perc_learnable_200': cp_200['perc_learnable'],
            'average_reward_200': cp_200['average_reward']
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def create_checkpoint_visualization(checkpoint_results, trajectory_df):
    """Create comprehensive checkpoint analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: R² progression
    ax1 = axes[0, 0]
    ax1.plot(checkpoint_results['percent_training'], checkpoint_results['r2'], 
             'o-', linewidth=3, markersize=10, color='#2E86AB')
    ax1.fill_between(checkpoint_results['percent_training'], 0, checkpoint_results['r2'], 
                     alpha=0.2, color='#2E86AB')
    
    # Add annotations for key checkpoints
    for _, row in checkpoint_results.iterrows():
        if row['checkpoint'] in [100, 200]:
            ax1.annotate(f"{row['r2']:.3f}", 
                        xy=(row['percent_training'], row['r2']),
                        xytext=(row['percent_training'] + 2, row['r2'] - 0.05),
                        fontsize=10)
    
    ax1.set_xlabel('Percent of Training Completed')
    ax1.set_ylabel('R² (Predictive Power)')
    ax1.set_title('Checkpoint Predictive Power')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Early vs late improvement
    ax2 = axes[0, 1]
    ax2.scatter(trajectory_df['improve_0_200'], trajectory_df['improve_200_1000'], 
                alpha=0.7, s=50)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Improvement 0→200')
    ax2.set_ylabel('Improvement 200→1000')
    ax2.set_title('Early vs Late Phase Learning')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation
    r, p = pearsonr(trajectory_df['improve_0_200'], trajectory_df['improve_200_1000'])
    ax2.text(0.05, 0.95, f'r = {r:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Error@100 vs Error@200 vs Final
    ax3 = axes[1, 0]
    
    # Error@100 vs Final
    ax3.scatter(trajectory_df['error_100'], trajectory_df['final_error'], 
                alpha=0.5, label='Error@100', s=30)
    
    # Error@200 vs Final
    ax3.scatter(trajectory_df['error_200'], trajectory_df['final_error'], 
                alpha=0.5, label='Error@200', s=30)
    
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('Early Checkpoint Error')
    ax3.set_ylabel('Final Error')
    ax3.set_title('Early Error Predictiveness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    mean_improve_early = trajectory_df['improve_0_200'].mean()
    std_improve_early = trajectory_df['improve_0_200'].std()
    mean_improve_late = trajectory_df['improve_200_1000'].mean()
    std_improve_late = trajectory_df['improve_200_1000'].std()
    
    summary_text = f"""Key Statistics:

Early Phase (0→200):
  Mean improvement: {mean_improve_early:.3f} ± {std_improve_early:.3f}
  
Late Phase (200→1000):
  Mean improvement: {mean_improve_late:.3f} ± {std_improve_late:.3f}
  
Predictive Power:
  Error@100: R² = {checkpoint_results[checkpoint_results['checkpoint']==100]['r2'].values[0]:.3f}
  Error@200: R² = {checkpoint_results[checkpoint_results['checkpoint']==200]['r2'].values[0]:.3f}
  
Key Insight:
  {checkpoint_results[checkpoint_results['checkpoint']==200]['r2'].values[0]*100:.0f}% of final performance 
  is determined by checkpoint 200!
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Early Checkpoint Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../scaling_analysis_results.csv')
    
    # Initialize predictor
    predictor = EarlyCheckpointPredictor()
    
    # Analyze all checkpoints
    checkpoint_results = predictor.analyze_all_checkpoints(df)
    print("Checkpoint Analysis:")
    print(checkpoint_results)
    
    # Extract trajectory features
    trajectory_df = analyze_trajectory_features(df)
    print(f"\nTrajectory features extracted for {len(trajectory_df)} runs")
    
    # Example prediction
    error_200 = 0.3
    final_pred = predictor.predict_from_checkpoint(error_200, 200)
    print(f"\nPrediction: Error@200 = {error_200} → Final error = {final_pred:.3f}")
