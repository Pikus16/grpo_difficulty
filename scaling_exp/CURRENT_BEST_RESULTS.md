# Current Best Scaling Law Results

## Best Scaling Law Found

After extensive experimentation, the optimal scaling law is:

```
error = 0.214 × model_size^(-0.529) × base^(-0.905) × perc_learnable^(-0.359)
```

**Performance**: R² = 0.669

## Key Findings

1. **Base accuracy** is the most important predictor (exponent ≈ -0.9)
2. **Model size** optimal exponent is -0.529 (not -0.577 as originally fitted)
3. **Percentage learnable** significantly improves predictions (exponent ≈ -0.36)
4. **Train score** has negligible effect and can be dropped

## Dataset Performance

- **GSM8K**: R² ≈ 0.42 (works well)
- **ShuffleObj**: R² ≈ 0.89 (works very well)
- **KEGG**: R² ≈ -28 (fails - fundamentally different dynamics)

## Original vs Optimized

| Version | Equation | R² |
|---------|----------|-----|
| Original (no perc_learnable) | error = 0.400 × model_size^(-0.577) × base^(-0.905) | 0.479 |
| Enhanced (with perc_learnable) | error = 0.214 × model_size^(-0.577) × base^(-0.905) × perc_learnable^(-0.359) | 0.647 |
| **Optimized** | **error = 0.214 × model_size^(-0.529) × base^(-0.905) × perc_learnable^(-0.359)** | **0.669** |

## Usage

This scaling law predicts final error rate given:
- `model_size`: Model parameters in billions
- `base`: Initial accuracy before training
- `perc_learnable`: Fraction of training problems showing improvement

## Data

Raw experimental data is in `scaling_analysis_results.csv` (480 datapoints across 48 runs)
