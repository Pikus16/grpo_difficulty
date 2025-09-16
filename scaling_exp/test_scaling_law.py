#!/usr/bin/env python3
"""
Simple script to test the current best optimized scaling law
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('scaling_analysis_results.csv')
final_df = df[df['checkpoint'] == 1000].copy()
final_df['error'] = 1 - final_df['final_acc']

# Optimized scaling law
def scaling_law(model_size, base, perc_learnable):
    """
    error = 0.214 × model_size^(-0.529) × base^(-0.905) × perc_learnable^(-0.359)
    """
    return 0.214 * (model_size ** -0.529) * (base ** -0.905) * (perc_learnable ** -0.359)

# Calculate predictions
predictions = scaling_law(
    final_df['model_size'].values,
    final_df['base'].values,
    final_df['perc_learnable'].values
)

# Calculate R²
r2 = r2_score(final_df['error'].values, predictions)

print("Optimized Scaling Law Test")
print("="*40)
print("error = 0.214 × model_size^(-0.529) × base^(-0.905) × perc_learnable^(-0.359)")
print(f"\nOverall R² = {r2:.4f}")

# Per-dataset R²
print("\nPer-dataset R²:")
for dataset in final_df['dataset'].unique():
    mask = final_df['dataset'] == dataset
    if mask.sum() > 0:
        dataset_r2 = r2_score(
            final_df.loc[mask, 'error'].values,
            predictions[mask]
        )
        print(f"  {dataset}: {dataset_r2:.4f}")

# Example predictions
print("\nExample predictions:")
print("Model Size | Base Acc | Perc Learn | Predicted Error | Actual Error")
print("-"*65)
for i in range(5):
    row = final_df.iloc[i]
    pred = predictions[i]
    print(f"{row['model_size']:>10.0f}B | {row['base']:>8.3f} | {row['perc_learnable']:>10.3f} | "
          f"{pred:>15.3f} | {row['error']:>12.3f}")
