# How to Enhance Cross-Validation Accuracy

Current Performance: **72-76%**
Target: **75-80%**

## 🎯 Top 5 Enhancement Techniques (Ranked by Impact)

### 1. **Increase Number of Trees** ⭐⭐⭐⭐⭐
**Expected Gain: +2-3%**

```python
# Current
RandomForestClassifier(n_estimators=100)

# Enhanced
RandomForestClassifier(n_estimators=500)  # or 1000
```

**Why it works**: More trees = more stable predictions = better generalization

---

### 2. **Regularization (Reduce Overfitting)** ⭐⭐⭐⭐⭐
**Expected Gain: +3-5%**

```python
# Current
RandomForestClassifier(
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1
)

# Enhanced
RandomForestClassifier(
    max_depth=15,              # Shallower trees
    min_samples_split=10,      # Need more samples to split
    min_samples_leaf=4         # Each leaf needs 4+ samples
)
```

**Why it works**: Your models currently show 27-32% gap between train and validation (overfitting). Regularization fixes this.

---

### 3. **Try Gradient Boosting** ⭐⭐⭐⭐
**Expected Gain: +2-4%**

```python
from sklearn.ensemble import GradientBoostingClassifier

GradientBoostingClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_samples_split=10
)
```

**Why it works**: GradientBoosting often performs better than RandomForest on smaller datasets

---

### 4. **Feature Engineering** ⭐⭐⭐
**Expected Gain: +1-2%**

Add derived features that capture domain knowledge:

```python
# Add these to your preprocessing:

# 1. Planet temperature bins
X['temp_category'] = pd.cut(X['koi_teq'], bins=[0, 200, 400, 1000, 5000], labels=[0,1,2,3])

# 2. Orbital period bins (hot jupiters vs earth-like)
X['period_category'] = pd.cut(X['koi_period'], bins=[0, 10, 100, 365, 10000], labels=[0,1,2,3])

# 3. Radius ratio (planet/star)
X['radius_ratio_squared'] = X['planet_star_radius_ratio'] ** 2

# 4. Habitability score
X['habitable_score'] = (
    (X['koi_teq'] > 200) & (X['koi_teq'] < 350) &  # Right temperature
    (X['koi_prad'] > 0.5) & (X['koi_prad'] < 2.0)   # Right size
).astype(int)
```

---

### 5. **Optimize max_features** ⭐⭐⭐
**Expected Gain: +1-2%**

```python
# Current (default)
RandomForestClassifier(max_features='sqrt')  # Uses sqrt(17) ≈ 4 features

# Try these:
RandomForestClassifier(max_features='log2')  # Uses log2(17) ≈ 4 features
RandomForestClassifier(max_features=0.5)     # Uses 50% = 8-9 features
```

**Why it works**: Reducing features per split increases tree diversity

---

## 📊 Quick Test (Run This)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Your current model
current_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Enhanced model
enhanced_model = RandomForestClassifier(
    n_estimators=500,          # More trees
    max_depth=15,              # Shallower (less overfitting)
    min_samples_split=10,      # More regularization
    min_samples_leaf=4,        # More regularization
    max_features='log2',       # Feature diversity
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

current_scores = cross_val_score(current_model, X, y, cv=cv, scoring='accuracy')
enhanced_scores = cross_val_score(enhanced_model, X, y, cv=cv, scoring='accuracy')

print(f"Current:  {current_scores.mean()*100:.2f}%")
print(f"Enhanced: {enhanced_scores.mean()*100:.2f}%")
print(f"Gain:     {(enhanced_scores.mean() - current_scores.mean())*100:+.2f}%")
```

---

## 🚀 Advanced Techniques (If Basic Doesn't Work)

### 6. **XGBoost** (Best Performance)
```bash
pip install xgboost
```

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```
Expected: **+5-8% improvement**

### 7. **Ensemble Voting**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=15)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=10)),
    ],
    voting='soft'
)
```
Expected: **+2-3% improvement**

### 8. **Collect More Data**
- Current: 6,484 samples (Cumulative), 3,047 (K2), 7,110 (TOI)
- NASA updates datasets regularly - check for new confirmed exoplanets
- Expected: **+3-5% with 2x more data**

---

## ⚠️ What WON'T Work

❌ **Neural Networks**: Too small dataset, will underperform
❌ **Deep Learning**: Same reason - need 100k+ samples
❌ **SMOTE oversampling**: Classes are already balanced
❌ **Removing features**: You already did feature selection

---

## 📈 Realistic Expectations

| Current | With Basic Enhancements | With XGBoost | With More Data |
|---------|------------------------|--------------|----------------|
| 72-76% | 75-80% | 78-83% | 80-85% |

**Most Important**: The #1 way to improve is **reduce overfitting**. Your train accuracy is 98% but validation is 72% - that's a 26% gap! Fix that first.

---

## 🎓 Summary

**Priority Order**:
1. ✅ Add regularization (min_samples_split=10, max_depth=15)
2. ✅ Increase trees (n_estimators=500)
3. ✅ Try Gradient Boosting
4. ✅ Install and test XGBoost
5. ⭐ Add domain-specific feature engineering

**Expected Total Improvement**: +5-10% accuracy
**New Expected Accuracy**: 77-82% cross-validation
