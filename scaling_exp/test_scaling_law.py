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


# -------------------------------------------------
# Held-out test set evaluation
# -------------------------------------------------
try:
    test_df = pd.read_csv('held_out_scaling_numbers.csv')
    test_df = test_df[test_df['checkpoint'] == 1000].copy()
    test_df['error'] = 1 - test_df['final_acc']

    test_predictions = scaling_law(
        test_df['model_size'].values,
        test_df['base'].values,
        test_df['perc_learnable'].values
    )

    test_r2 = r2_score(test_df['error'].values, test_predictions)

    print("\nHeld-out Test Set Evaluation")
    print("="*40)
    print(f"Overall R² (test) = {test_r2:.4f}")

    print("\nPer-dataset R² (test):")
    for dataset in test_df['dataset'].unique():
        mask = test_df['dataset'] == dataset
        if mask.sum() > 0:
            dataset_r2 = r2_score(
                test_df.loc[mask, 'error'].values,
                test_predictions[mask]
            )
            print(f"  {dataset}: {dataset_r2:.4f}")

    print("\nExample predictions (test):")
    print("Model Size | Base Acc | Perc Learn | Predicted Error | Actual Error")
    print("-"*65)
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        pred = test_predictions[i]
        print(f"{row['model_size']:>10.0f}B | {row['base']:>8.3f} | {row['perc_learnable']:>10.3f} | "
              f"{pred:>15.3f} | {row['error']:>12.3f}")

except FileNotFoundError:
    print("\nNo held-out test set CSV found (expected 'scaling_analysis_test.csv'). Skipping test evaluation.")