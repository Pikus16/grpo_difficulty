#!/usr/bin/env python3
"""
Run all scaling law experiments and evaluations
Consolidated from multiple experiment files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.special import logit, expit
from scaling_law_models import ScalingLawModels, predict_final_error

def load_data():
    """Load training and held-out datasets"""
    train_df = pd.read_csv('scaling_analysis_results.csv')
    held_out_df = pd.read_csv('held_out_scaling_numbers.csv')
    return train_df, held_out_df

def evaluate_all_models(train_df, held_out_df):
    """Evaluate all model variants on both datasets"""
    models = ScalingLawModels()
    results = []
    
    # Get final checkpoint data
    train_final = train_df[train_df['checkpoint'] == 1000].copy()
    held_out_final = held_out_df[held_out_df['checkpoint'] == 1000].copy()
    
    # Add derived features
    for df in [train_final, held_out_final]:
        df['error'] = 1 - df['final_acc']
        df['log_model_size'] = np.log(df['model_size'])
    
    # Model 1: Basic power law
    train_pred_basic = train_final.apply(
        lambda r: models.basic_power_law(r['model_size'], r['base']), axis=1)
    train_r2_basic = r2_score(train_final['error'], train_pred_basic)
    
    results.append({
        'Model': 'Basic Power Law',
        'Training R²': train_r2_basic,
        'Held-out R²': None,
        'Parameters': 3
    })
    
    # Model 2: With percentage learnable
    train_pred_perc = train_final.apply(
        lambda r: models.power_law_with_learnable(
            r['model_size'], r['base'], r['perc_learnable']), axis=1)
    train_r2_perc = r2_score(train_final['error'], train_pred_perc)
    
    results.append({
        'Model': '+ Percentage Learnable',
        'Training R²': train_r2_perc,
        'Held-out R²': None,
        'Parameters': 4
    })
    
    # Model 3: Logit model
    train_pred_logit = train_final.apply(
        lambda r: models.logit_model(
            r['model_size'], r['base'], r['perc_learnable']), axis=1)
    train_r2_logit = r2_score(train_final['error'], train_pred_logit)
    
    results.append({
        'Model': 'Logit Transformation',
        'Training R²': train_r2_logit,
        'Held-out R²': None,
        'Parameters': 4
    })
    
    # Model 4: Fixed effects
    train_pred_fixed = train_final.apply(
        lambda r: models.fixed_effects_model(
            r['model_size'], r['base'], r['perc_learnable'],
            r['dataset'], r['strategy']), axis=1)
    train_r2_fixed = r2_score(train_final['error'], train_pred_fixed)
    
    results.append({
        'Model': '+ Fixed Effects',
        'Training R²': train_r2_fixed,
        'Held-out R²': None,
        'Parameters': 10
    })
    
    # Model 5: Early checkpoints (if available)
    # Get checkpoint 200 data
    train_200 = train_df[train_df['checkpoint'] == 200][
        ['dataset', 'strategy', 'model_name', 'accuracy']].copy()
    train_200['error_200'] = 1 - train_200['accuracy']
    
    # Merge with final
    train_merged = train_final.merge(
        train_200[['dataset', 'strategy', 'model_name', 'error_200']], 
        on=['dataset', 'strategy', 'model_name'])
    
    if len(train_merged) > 0:
        train_pred_cp200 = train_merged['error_200'].apply(
            models.predict_from_checkpoint_200)
        train_r2_cp200 = r2_score(train_merged['error'], train_pred_cp200)
        
        # Try on held-out
        held_200 = held_out_df[held_out_df['checkpoint'] == 200][
            ['dataset', 'strategy', 'model_name', 'accuracy']].copy()
        held_200['error_200'] = 1 - held_200['accuracy']
        
        held_merged = held_out_final.merge(
            held_200[['dataset', 'strategy', 'model_name', 'error_200']], 
            on=['dataset', 'strategy', 'model_name'])
        
        if len(held_merged) > 0:
            held_pred_cp200 = held_merged['error_200'].apply(
                models.predict_from_checkpoint_200)
            held_r2_cp200 = r2_score(held_merged['error'], held_pred_cp200)
        else:
            held_r2_cp200 = None
            
        results.append({
            'Model': 'Error@200 Only',
            'Training R²': train_r2_cp200,
            'Held-out R²': held_r2_cp200,
            'Parameters': 2
        })
    
    return pd.DataFrame(results)

def analyze_early_checkpoints(train_df):
    """Analyze predictive power at different checkpoints"""
    checkpoints = [100, 200, 300, 500]
    results = []
    
    # Get final data
    final_df = train_df[train_df['checkpoint'] == 1000][
        ['dataset', 'strategy', 'model_name', 'final_acc']].copy()
    final_df['final_error'] = 1 - final_df['final_acc']
    
    for cp in checkpoints:
        cp_df = train_df[train_df['checkpoint'] == cp][
            ['dataset', 'strategy', 'model_name', 'accuracy']].copy()
        cp_df[f'error_{cp}'] = 1 - cp_df['accuracy']
        
        # Merge with final
        merged = final_df.merge(cp_df, on=['dataset', 'strategy', 'model_name'])
        
        if len(merged) > 0:
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            X = merged[f'error_{cp}'].values.reshape(-1, 1)
            y = merged['final_error'].values
            lr.fit(X, y)
            r2 = lr.score(X, y)
            
            results.append({
                'Checkpoint': cp,
                'Percent_Training': cp / 10,
                'R²': r2,
                'N_samples': len(merged),
                'Slope': lr.coef_[0],
                'Intercept': lr.intercept_
            })
    
    return pd.DataFrame(results)

def create_summary_visualization(results_df, checkpoint_df):
    """Create comprehensive visualization of all results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Model Evolution
    ax1 = axes[0, 0]
    models_to_plot = results_df[results_df['Training R²'].notna()]
    x = np.arange(len(models_to_plot))
    bars = ax1.bar(x, models_to_plot['Training R²'], alpha=0.8)
    
    # Color code by R² value
    colors = plt.cm.RdYlGn(models_to_plot['Training R²'].values)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_to_plot['Model'], rotation=45, ha='right')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Evolution of Scaling Law Models')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, models_to_plot['Training R²']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 2: Early Checkpoint Predictiveness
    ax2 = axes[0, 1]
    ax2.plot(checkpoint_df['Percent_Training'], checkpoint_df['R²'], 
             'o-', linewidth=3, markersize=10, color='#2E86AB')
    ax2.fill_between(checkpoint_df['Percent_Training'], 0, checkpoint_df['R²'], 
                     alpha=0.2, color='#2E86AB')
    
    ax2.set_xlabel('Percent of Training Completed')
    ax2.set_ylabel('R² (Predictive Power)')
    ax2.set_title('Early Checkpoint Predictive Power')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add annotations
    for _, row in checkpoint_df.iterrows():
        if row['Checkpoint'] in [100, 200]:
            ax2.annotate(f"{row['R²']:.3f}", 
                        xy=(row['Percent_Training'], row['R²']),
                        xytext=(row['Percent_Training'] + 2, row['R²'] - 0.05),
                        fontsize=10)
    
    # Plot 3: Parameter Efficiency
    ax3 = axes[1, 0]
    valid_models = results_df[results_df['Training R²'].notna()]
    ax3.scatter(valid_models['Parameters'], valid_models['Training R²'], 
                s=100, alpha=0.7)
    
    # Add labels
    for _, row in valid_models.iterrows():
        if row['Training R²'] > 0.7:  # Only label good models
            ax3.annotate(row['Model'], 
                        xy=(row['Parameters'], row['Training R²']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('Training R²')
    ax3.set_title('Model Complexity vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Key Insights Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insights = [
        "Key Findings:",
        "",
        f"• Best Training R²: {results_df['Training R²'].max():.3f}",
        f"• Error@200 R²: {checkpoint_df[checkpoint_df['Checkpoint']==200]['R²'].values[0]:.3f}",
        f"• Error@100 R²: {checkpoint_df[checkpoint_df['Checkpoint']==100]['R²'].values[0]:.3f}",
        "",
        "Practical Implications:",
        "• 75% predictable by checkpoint 200",
        "• 53% predictable by checkpoint 100", 
        "• Early stopping saves 80-90% compute",
        "",
        "Best Practices:",
        "• Always log checkpoints 100, 200",
        "• Use error@200 for decisions",
        "• Screen widely, filter early"
    ]
    
    y_pos = 0.95
    for line in insights:
        weight = 'bold' if line.endswith(':') else 'normal'
        ax4.text(0.05, y_pos, line, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', weight=weight)
        y_pos -= 0.07
    
    plt.suptitle('GRPO Scaling Laws: Complete Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scaling_law_summary.png', dpi=150, bbox_inches='tight')
    print("Saved comprehensive summary to scaling_law_summary.png")

def generate_latex_formulas():
    """Generate LaTeX formulas for the paper"""
    formulas = {
        'Basic Power Law': r'$\mathcal{E} = 0.214 \times M^{-0.577} \times B^{-0.905}$',
        'With Percentage Learnable': r'$\mathcal{E} = 0.214 \times M^{-0.577} \times B^{-0.905} \times L^{-0.359}$',
        'Logit Model': r'$\text{logit}(\mathcal{E}) = -2.555 - 0.752\log(M) - 2.197\log(B) - 0.776\log(L)$',
        'Early Trajectory': r'$\text{logit}(\mathcal{E}_1) = \text{logit}(\mathcal{E}_0) - 0.532 - 0.399\log(M) - 0.286\text{logit}(L) + 147.4 S_{0:200}$',
        'Simple Checkpoint': r'$\mathcal{E}_{\text{final}} = -0.091 + 0.828 \times \mathcal{E}_{100}$'
    }
    
    print("\n" + "="*70)
    print("LATEX FORMULAS FOR PAPER")
    print("="*70)
    for name, formula in formulas.items():
        print(f"\n{name}:")
        print(formula)
    
    return formulas

# Main execution
if __name__ == "__main__":
    print("Running comprehensive scaling law analysis...")
    
    # Load data
    train_df, held_out_df = load_data()
    print(f"Loaded {len(train_df)} training points, {len(held_out_df)} held-out points")
    
    # Evaluate all models
    print("\nEvaluating all model variants...")
    results_df = evaluate_all_models(train_df, held_out_df)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    
    # Analyze checkpoints
    print("\nAnalyzing early checkpoint predictiveness...")
    checkpoint_df = analyze_early_checkpoints(train_df)
    print("\nCheckpoint Analysis:")
    print(checkpoint_df.to_string(index=False))
    
    # Create visualizations
    print("\nCreating summary visualizations...")
    create_summary_visualization(results_df, checkpoint_df)
    
    # Generate LaTeX
    formulas = generate_latex_formulas()
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    checkpoint_df.to_csv('checkpoint_analysis_results.csv', index=False)
    
    print("\n✅ Analysis complete! Results saved to:")
    print("   - scaling_law_summary.png")
    print("   - model_comparison_results.csv")
    print("   - checkpoint_analysis_results.csv")
