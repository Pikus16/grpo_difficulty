# Strategy 3: Evaluation Results

**Split:** 54 train runs / 61 test runs  
**Date:** October 16, 2025

---

## Executive Summary

### Top Performers (Held-Out R²):

1. **CP200 Logit** - R² = 0.814 ⭐⭐⭐ BEST
2. **Trajectory CP200** - R² = 0.803 ⭐⭐ RECOMMENDED
3. **CP100 Logit** - R² = 0.602 ⭐ EARLY
4. **Trajectory CP100** - R² = 0.547 ⚠️

---

## Table 1: Detailed Performance Metrics

| Model | R² Train | R² Held-Out | N Train | N Held | Calib Slope | Eligibility | Status |
|-------|----------|-------------|---------|--------|-------------|-------------|--------|
| CP100 Logit | 0.6040 | 0.6022 | 56 | 39 | 1.0189 ✅ | 39/61 (64%) | ⭐ Good |
| CP200 Logit | 0.7862 | 0.8140 | 56 | 61 | 1.0949 ✅ | 61/61 (100%) | ⭐⭐⭐ BEST |
| Trajectory CP100 | 0.8067 | 0.5465 | 54 | 39 | 0.8849 ⚠️ | 39/61 (64%) | ⚠️ Fair |
| Trajectory CP200 | 0.8760 | 0.8033 | 54 | 61 | 0.9516 ✅ | 61/61 (100%) | ⭐⭐⭐ BEST |

**Calibration Notes:**
- Target: 0.9-1.1 (slope=1.0 means perfectly calibrated)
- CP200 Logit: 1.095 ✅ Excellent
- CP100 Logit: 1.019 ✅ Excellent
- Trajectory CP200: 0.952 ✅ Excellent
- Trajectory CP100: 0.885 ⚠️ Fair

---

## Table 2: Policy Savings - Top 2 Models


### CP200 Logit

| Threshold | Compute Saved | Winners Missed |
|-----------|---------------|----------------|
| 5pp | 28.9% | 2.8% |
| 10pp | 28.9% | 0.0% |
| 15pp | 31.5% | 3.2% |
| 20pp | 43.3% | 3.6% |

### Trajectory CP200

| Threshold | Compute Saved | Winners Missed |
|-----------|---------------|----------------|
| 5pp | 27.5% | 2.8% |
| 10pp | 27.5% | 2.9% |
| 15pp | 27.5% | 3.2% |
| 20pp | 35.4% | 0.0% |

---

## Table 3: Cross-Validation Results

**What is Cross-Validation?**
Cross-validation splits the training data into 5 folds, trains on 4 folds and validates on the remaining fold, repeating for all folds. This tests if the model generalizes well even within the training set and helps detect overfitting.

**Interpreting CV vs Held-Out:**
- **CV ≈ Held-Out:** Model generalizes well ✅
- **CV >> Held-Out:** Model may overfit to training distribution ⚠️
- **CV << Held-Out:** Test set is easier than expected ⚠️

| Model | CV R² (5-fold) | Held-Out R² | Difference | Assessment |
|-------|----------------|-------------|------------|------------|
| CP100 Logit | 0.2201±0.926 | 0.6022 | -0.3821 | ⚠️ Test easier |
| CP200 Logit | 0.5645±0.604 | 0.8140 | -0.2496 | ⚠️ Test easier |
| Trajectory CP100 | 0.6215±0.272 | 0.5465 | +0.0750 | ✓ Acceptable |
| Trajectory CP200 | 0.7685±0.213 | 0.8033 | -0.0347 | ✅ Good match |

**Validation Notes:**
- **CP100 Logit:** Held-out >> CV (-0.382) suggests test set is easier or different distribution than training.
- **CP200 Logit:** Held-out >> CV (-0.250) suggests test set is easier or different distribution than training.
- **Trajectory CP200:** Excellent match (Δ=-0.035), results are reliable.

---

## Status

✅ **EVALUATION COMPLETE**

- Perfect reproducibility (std = 0.000 across 3 runs)
- No data leakage
- All 4 models evaluated
- Policy savings computed
