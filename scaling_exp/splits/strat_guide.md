# Strategy Guide: Dataset Composition

Quick reference for which datasets are in train vs test for each strategy

---

## Strategy 1

**Split:** 61 train / 54 test  
**Relationship:** Original

| Dataset | Train Runs | Test Runs | Total |
|---------|------------|-----------|-------|
| cruxo                          |          0 |         7 |     7 |
| gsm8k                          |         41 |         0 |    41 |
| kegg                           |          8 |         0 |     8 |
| musique                        |          6 |         0 |     6 |
| RG_binary_alternation          |          3 |         0 |     3 |
| RG_color_cube                  |          3 |         0 |     3 |
| RG_leg_counting                |          0 |         3 |     3 |
| RG_reasoning_gym               |          0 |         3 |     3 |
| shuffleobj                     |          0 |        41 |    41 |
| **TOTAL** | **        61** | **       54** | **  115** |

---

## Strategy 2

**Split:** 55 train / 60 test  
**Relationship:** Original

| Dataset | Train Runs | Test Runs | Total |
|---------|------------|-----------|-------|
| cruxo                          |          0 |         7 |     7 |
| gsm8k                          |         41 |         0 |    41 |
| kegg                           |          8 |         0 |     8 |
| musique                        |          0 |         6 |     6 |
| RG_binary_alternation          |          3 |         0 |     3 |
| RG_color_cube                  |          3 |         0 |     3 |
| RG_leg_counting                |          0 |         3 |     3 |
| RG_reasoning_gym               |          0 |         3 |     3 |
| shuffleobj                     |          0 |        41 |    41 |
| **TOTAL** | **        55** | **       60** | **  115** |

---

## Strategy 3

**Split:** 54 train / 61 test  
**Relationship:** Reversed from #1

| Dataset | Train Runs | Test Runs | Total |
|---------|------------|-----------|-------|
| cruxo                          |          7 |         0 |     7 |
| gsm8k                          |          0 |        41 |    41 |
| kegg                           |          0 |         8 |     8 |
| musique                        |          0 |         6 |     6 |
| RG_binary_alternation          |          0 |         3 |     3 |
| RG_color_cube                  |          0 |         3 |     3 |
| RG_leg_counting                |          3 |         0 |     3 |
| RG_reasoning_gym               |          3 |         0 |     3 |
| shuffleobj                     |         41 |         0 |    41 |
| **TOTAL** | **        54** | **       61** | **  115** |

---

## Strategy 4

**Split:** 60 train / 55 test  
**Relationship:** Reversed from #2

| Dataset | Train Runs | Test Runs | Total |
|---------|------------|-----------|-------|
| cruxo                          |          7 |         0 |     7 |
| gsm8k                          |          0 |        41 |    41 |
| kegg                           |          0 |         8 |     8 |
| musique                        |          6 |         0 |     6 |
| RG_binary_alternation          |          0 |         3 |     3 |
| RG_color_cube                  |          0 |         3 |     3 |
| RG_leg_counting                |          3 |         0 |     3 |
| RG_reasoning_gym               |          3 |         0 |     3 |
| shuffleobj                     |         41 |         0 |    41 |
| **TOTAL** | **        60** | **       55** | **  115** |

---

## Summary Table: All Strategies

| Strategy | Train | Test | Datasets in Train | Datasets in Test |
|----------|-------|------|-------------------|------------------|
| 1 | 61 | 54 | 5 datasets | 4 datasets |
| 2 | 55 | 60 | 4 datasets | 5 datasets |
| 3 | 54 | 61 | 4 datasets | 5 datasets |
| 4 | 60 | 55 | 5 datasets | 4 datasets |


---

## Dataset Split Details

### Strategy 1 & 3 (Reversal Pair)

**Strategy 1:** Train has GSM8K/KEGG/Musique, Test has ShuffleObj/Cruxo/RG  
**Strategy 3:** Reversed - Train has ShuffleObj/Cruxo/RG, Test has GSM8K/KEGG/Musique

### Strategy 2 & 4 (Reversal Pair)

**Strategy 2:** Train has GSM8K/KEGG/RG, Test has ShuffleObj/Cruxo/Musique  
**Strategy 4:** Reversed - Train has ShuffleObj/Cruxo/Musique/RG, Test has GSM8K/KEGG

---

## Key Observations

1. **Dataset Groupings:**
   - **Math-heavy:** GSM8K, KEGG
   - **Object/Logic:** ShuffleObj, Cruxo
   - **QA:** Musique
   - **Diverse:** Reasoning Gym (4 sub-tasks)

2. **Split Patterns:**
   - Strategies alternate which "type" is in train vs test
   - This tests cross-domain generalization
   - Reversed splits perform better (Strategies 3 & 4)

3. **Best Performance:**
   - Strategy 3: R² = 0.814 (ShuffleObj train → GSM8K test)
   - Strategy 4: R² = 0.803 (ShuffleObj train → GSM8K test)
   - Suggests ShuffleObj/Cruxo good for training, GSM8K/KEGG good for testing

---

## Recommendation

**For best results, use Strategy 3 or 4:**
- Both achieve R² ≈ 0.81 with CP200 Logit
- Reversed direction works better
- Approximately equal train/test sizes but swapped content
