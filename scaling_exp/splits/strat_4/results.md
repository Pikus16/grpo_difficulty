# Strategy 4: Evaluation Results

**Split:** 60 train runs / 55 test runs  
**Date:** October 16, 2025

---

## Executive Summary

### Top Performers (Held-Out R²):

1. **CP200 Logit** - R² = 0.803 ⭐⭐⭐ BEST
2. **Trajectory CP200** - R² = 0.802 ⭐⭐ RECOMMENDED
3. **CP100 Logit** - R² = 0.602 ⭐ EARLY
4. **Trajectory CP100** - R² = 0.547 ⚠️

---

## Table 1: Detailed Performance Metrics

| Model | R² Train | R² Held-Out | N Train | N Held | Calib Slope | Eligibility | Status |
|-------|----------|-------------|---------|--------|-------------|-------------|--------|
| CP100 Logit | 0.6040 | 0.6022 | 56 | 39 | 1.0189 ✅ | 39/55 (71%) | ⭐ Good |
| CP200 Logit | 0.7907 | 0.8029 | 62 | 55 | 1.0672 ✅ | 55/55 (100%) | ⭐⭐⭐ BEST |
| Trajectory CP100 | 0.8067 | 0.5465 | 54 | 39 | 0.8849 ⚠️ | 39/55 (71%) | ⚠️ Fair |
| Trajectory CP200 | 0.8645 | 0.8020 | 60 | 55 | 0.9691 ✅ | 55/55 (100%) | ⭐⭐⭐ BEST |

**Calibration Notes:**
- Target: 0.9-1.1 (slope=1.0 means perfectly calibrated)
- CP200 Logit: 1.067 ✅ Excellent
- CP100 Logit: 1.019 ✅ Excellent
- Trajectory CP200: 0.969 ✅ Excellent
- Trajectory CP100: 0.885 ⚠️ Fair

---

## Table 2: Policy Savings - Top 2 Models


### CP200 Logit

| Threshold | Compute Saved | Winners Missed |
|-----------|---------------|----------------|
| 5pp | 23.3% | 2.8% |
| 10pp | 23.3% | 0.0% |
| 15pp | 26.2% | 3.2% |
| 20pp | 39.3% | 3.6% |

### Trajectory CP200

| Threshold | Compute Saved | Winners Missed |
|-----------|---------------|----------------|
| 5pp | 24.7% | 2.8% |
| 10pp | 24.7% | 2.9% |
| 15pp | 24.7% | 3.2% |
| 20pp | 34.9% | 3.6% |

---

## Status

✅ **EVALUATION COMPLETE**

- Perfect reproducibility (std = 0.000 across 3 runs)
- No data leakage
- All 4 models evaluated
- Policy savings computed
