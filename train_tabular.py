# train_tabular.py — improved version
# Key changes from previous version:
# 1. Adds SMOTE oversampling to balance classes better during training
# 2. Uses better XGBoost hyperparameters tuned for small datasets
# 3. Adds feature engineering — creates new meaningful features from existing ones
#    e.g. temp/humidity ratio is more predictive than either alone
# 4. Uses a larger portion for training since dataset is very small

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from preprocess import CSV_PATH, BASE

os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

MODEL_SAVE  = os.path.join(BASE, "models",  "xgb_fire.pkl")
SCALER_SAVE = os.path.join(BASE, "models",  "scaler.pkl")
SHAP_PLOT   = os.path.join(BASE, "outputs", "shap_summary.png")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = pd.get_dummies(df, columns=['month', 'day'])
df['label'] = (df['area'] > 0).astype(int)
df = df.drop(columns=['area'])

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
# We create new features by combining existing ones
# These combinations are often MORE predictive than raw features alone
# because they capture relationships the model might miss

# Heat stress index: high temp + low humidity = extreme fire danger
# When temp is high AND humidity is low, fire risk spikes dramatically
df['temp_humidity_ratio'] = df['temp'] / (df['RH'] + 1)

# Fire Weather Index components combined
# FFMC measures moisture of fine fuels — higher = drier = more fire risk
# ISI measures fire spread potential — higher = faster spread
df['ffmc_isi_product'] = df['FFMC'] * df['ISI']

# Drought + wind interaction
# DC (Drought Code) measures deep moisture — high DC + high wind = danger
df['dc_wind_interaction'] = df['DC'] * df['wind']

# Dryness score: combines all moisture-related features into one signal
df['dryness_score'] = df['FFMC'] + df['DMC'] + (df['DC'] / 10)

print("Feature engineering added 4 new features")

X = df.drop(columns=['label'])
y = df['label']
feature_names = list(X.columns)

print(f"Dataset shape  : {X.shape}")
print(f"Fire cases     : {y.sum()} ({100*y.mean():.1f}%)")
print(f"No-fire cases  : {(y==0).sum()} ({100*(1-y.mean()):.1f}%)\n")

# ── SCALE ─────────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use 85/15 split instead of 80/20 — gives model more data to learn from
# With only 517 rows, every training sample counts
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size   = 0.15,
    random_state= 42,
    stratify    = y
)

print(f"Train size : {len(X_train)} samples")
print(f"Test size  : {len(X_test)}  samples\n")

# ── SMOTE — synthetic oversampling ────────────────────────────────────────────
# SMOTE creates synthetic training samples by interpolating between
# existing samples of the minority class
# This helps the model learn the minority class better
# Install if needed: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — Train size: {len(X_train_bal)} "
          f"(fire={y_train_bal.sum()}, "
          f"nofire={(y_train_bal==0).sum()})")
except ImportError:
    # If imbalanced-learn not installed, just use original data
    print("imbalanced-learn not found — using original data")
    print("To install: pip install imbalanced-learn")
    X_train_bal, y_train_bal = X_train, y_train

# ── XGBOOST — tuned for small datasets ───────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators      = 300,
    max_depth         = 3,       # shallower trees = less overfit on small data
    learning_rate     = 0.03,    # slower learning = more careful, usually better
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_child_weight  = 5,       # higher = more conservative splits
    gamma             = 0.1,     # minimum loss reduction to make a split
                                  # higher = more conservative tree building
    reg_alpha         = 0.1,     # L1 regularization — penalizes complex models
    reg_lambda        = 1.5,     # L2 regularization — same purpose
    scale_pos_weight  = (y_train==0).sum() / (y_train==1).sum(),
                                  # balances class weights automatically
                                  # tells model fire cases are equally important
    eval_metric       = 'auc',
    random_state      = 42
)

# ── CROSS VALIDATION ──────────────────────────────────────────────────────────
print("Running 5-fold cross validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
print(f"Cross-val ROC-AUC : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Individual folds  : {[f'{s:.3f}' for s in cv_scores]}\n")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("Training final XGBoost model...")
model.fit(
    X_train_bal, y_train_bal,
    eval_set = [(X_test, y_test)],
    verbose  = 50
)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n=== TABULAR MODEL RESULTS ===")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"\nROC-AUC explanation:")
print(f"  0.5 = random guessing | 0.7 = okay | 0.8 = good | 0.9+ = excellent")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['No Fire', 'Fire']))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"                Predicted No Fire   Predicted Fire")
print(f"Actual No Fire  {cm[0][0]:^18}  {cm[0][1]:^14}")
print(f"Actual Fire     {cm[1][0]:^18}  {cm[1][1]:^14}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP feature importance plot...")
explainer = shap.TreeExplainer(model)
X_test_df = pd.DataFrame(
    scaler.inverse_transform(X_test),
    columns=feature_names
)
shap_values = explainer.shap_values(X_test_df)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test_df,
    feature_names=feature_names,
    show=False,
    max_display=15
)
plt.title("SHAP Feature Importance — Wildfire Risk Prediction", pad=15)
plt.tight_layout()
plt.savefig(SHAP_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"SHAP plot saved to: {SHAP_PLOT}")

pickle.dump(model,  open(MODEL_SAVE,  'wb'))
pickle.dump(scaler, open(SCALER_SAVE, 'wb'))

print(f"\nXGBoost saved to : {MODEL_SAVE}")
print(f"Scaler saved to  : {SCALER_SAVE}")
print("\nNext step: run train_fusion.py")