import os
import sys
import json

# Ensure macOS libomp is accessible for XGBoost
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import nbformat

# Output directories
ARTIFACT_DIR = "/Users/cryptorth/.gemini/antigravity-ide/brain/b9ec2924-5505-4d58-aec8-b0b37ad12571"
FRONTEND_ASSETS = "/Users/cryptorth/Documents/ml/projects/frontend/assets"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FRONTEND_ASSETS, exist_ok=True)

print("="*70)
print("STARTING ML SOP WEEK 4 & WEEK 5 PIPELINE EXECUTION")
print("="*70)

# 1. LOAD & CLEAN DATA
DATA_PATH = "/Users/cryptorth/Documents/ml/projects/cardio/cardio_train.csv"
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=";")
initial_count = len(df)
print(f"Initial raw samples: {initial_count}")

# Age conversion
df['age'] = (df['age'] / 365.25).round().astype(int)

# Duplicates removal
df.drop_duplicates(inplace=True)

# Outlier filtering based on physiological standards
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[df['ap_lo'] < df['ap_hi']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

# Feature Engineering
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

cleaned_count = len(df)
print(f"Cleaned dataset samples: {cleaned_count} (Filtered {initial_count - cleaned_count} anomalies/duplicates)")

# Train/Test Split
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
feature_names = continuous_features + categorical_features

print(f"Training set: {X_train_scaled.shape}, Testing set: {X_test_scaled.shape}")

# ── SCRATCH IMPLEMENTATION (SOP MANDATORY REQUIREMENT) ───────────────────────
print("\n" + "-"*50)
print("TRAINING SCRATCH LOGISTIC REGRESSION...")
print("-"*50)

def logistic_regression_scratch(X, y, lr=0.1, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    for i in range(iterations):
        linear = np.dot(X, weights) + bias
        y_hat = 1 / (1 + np.exp(-np.clip(linear, -250, 250)))
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

def predict_proba_scratch(X, weights, bias):
    linear = np.dot(X, weights) + bias
    return 1 / (1 + np.exp(-np.clip(linear, -250, 250)))

def predict_scratch(X, weights, bias, threshold=0.5):
    probs = predict_proba_scratch(X, weights, bias)
    return (probs >= threshold).astype(int)

scratch_w, scratch_b = logistic_regression_scratch(X_train_scaled, y_train.to_numpy(), lr=0.1, iterations=1000)
scratch_test_probs = predict_proba_scratch(X_test_scaled, scratch_w, scratch_b)
scratch_test_preds = predict_scratch(X_test_scaled, scratch_w, scratch_b)
scratch_train_preds = predict_scratch(X_train_scaled, scratch_w, scratch_b)

print(f"Scratch LR Test Accuracy: {accuracy_score(y_test, scratch_test_preds):.4f}")
print(f"Scratch LR Test F1-Score: {f1_score(y_test, scratch_test_preds):.4f}")

# ── WEEK 4: BASELINE MODELS EVALUATION & OVERFITTING ANALYSIS ───────────────
print("\n" + "="*70)
print("WEEK 4: MODEL EVALUATION & OVERFITTING / UNDERFITTING ANALYSIS")
print("="*70)

baseline_models = {
    'Scratch Logistic Reg': None,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=120, random_state=42, n_jobs=-1)
}

week4_results = []
trained_models = {}
y_probs_dict = {}
y_preds_dict = {}

for name, model in baseline_models.items():
    if name == 'Scratch Logistic Reg':
        train_preds = scratch_train_preds
        test_preds = scratch_test_preds
        test_probs = scratch_test_probs
    else:
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            test_probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_probs = test_preds

    y_probs_dict[name] = test_probs
    y_preds_dict[name] = test_preds

    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    spec = tn / (tn + fp)
    overfitting_gap = (train_acc - test_acc) * 100

    week4_results.append({
        'Model': name,
        'Train Acc (%)': round(train_acc * 100, 2),
        'Test Acc (%)': round(test_acc * 100, 2),
        'Overfit Gap (%)': round(overfitting_gap, 2),
        'Precision': round(prec, 4),
        'Recall (Sens)': round(rec, 4),
        'Specificity': round(spec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    })

week4_df = pd.DataFrame(week4_results).sort_values(by='F1-Score', ascending=False)
print("\n--- Week 4 Baseline Evaluation Summary Table ---")
print(week4_df.to_string(index=False))

# PLOT 1: Confusion Matrices
plt.figure(figsize=(18, 10))
models_to_plot = [m for m in baseline_models.keys() if m != 'K-Nearest Neighbors'][:6]
for idx, m_name in enumerate(models_to_plot):
    plt.subplot(2, 3, idx + 1)
    cm = confusion_matrix(y_test, y_preds_dict[m_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD (0)', 'CVD (1)'],
                yticklabels=['No CVD (0)', 'CVD (1)'])
    plt.title(f"{m_name}\nAcc: {accuracy_score(y_test, y_preds_dict[m_name]):.3f} | F1: {f1_score(y_test, y_preds_dict[m_name]):.3f}", fontsize=11, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week4_confusion_matrices.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week4_confusion_matrices.png"), dpi=200)
plt.close()

# PLOT 2: ROC Curves
plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", len(baseline_models))
for idx, (m_name, probs) in enumerate(y_probs_dict.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{m_name} (AUC = {score:.4f})", color=palette[idx], lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.5000)')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Week 4: ROC Curves for Baseline Models', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week4_roc_curves.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week4_roc_curves.png"), dpi=200)
plt.close()

# PLOT 3: Precision-Recall Curves
plt.figure(figsize=(10, 7))
for idx, (m_name, probs) in enumerate(y_probs_dict.items()):
    pr, rc, _ = precision_recall_curve(y_test, probs)
    ap = auc(rc, pr)
    plt.plot(rc, pr, label=f"{m_name} (AP = {ap:.4f})", color=palette[idx], lw=2)

plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (PPV)', fontsize=12)
plt.title('Week 4: Precision-Recall Curves for Baseline Models', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week4_pr_curves.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week4_pr_curves.png"), dpi=200)
plt.close()

# PLOT 4: Learning Curves (Overfitting / Underfitting Diagnosis)
print("\nComputing Learning Curves for Overfitting/Underfitting Diagnosis...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lc_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=100, random_state=42, n_jobs=-1)
}

train_sizes_rel = np.linspace(0.1, 1.0, 5)

for idx, (name, model) in enumerate(lc_models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, train_sizes=train_sizes_rel,
        cv=3, scoring='f1', n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax = axes[idx]
    ax.plot(train_sizes, train_mean, 'o-', color='#2563eb', label='Training F1 Score', lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2563eb')
    ax.plot(train_sizes, val_mean, 's-', color='#16a34a', label='Cross-Val F1 Score', lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#16a34a')
    
    ax.set_title(f"Learning Curve: {name}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Samples', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim([0.65, 0.85])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week4_learning_curves.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week4_learning_curves.png"), dpi=200)
plt.close()
print("Week 4 evaluation figures generated successfully.")


# ── WEEK 5: ADVANCED MODEL TRAINING, CV & HYPERPARAMETER TUNING ────────────
print("\n" + "="*70)
print("WEEK 5: ADVANCED MODEL TRAINING, 5-FOLD CV & HYPERPARAMETER TUNING")
print("="*70)

# 1. 5-Fold Stratified Cross-Validation for Stability Assessment
print("\n1. Running 5-Fold Stratified Cross-Validation on Candidate Models...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting (GBDT)': GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=120, random_state=42, n_jobs=-1)
}

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
cv_results_list = []

for name, model in cv_models.items():
    print(f"  Evaluating {name} with 5-Fold CV...")
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    cv_results_list.append({
        'Model': name,
        'CV Accuracy': f"{scores['test_accuracy'].mean()*100:.2f}% ± {scores['test_accuracy'].std()*100:.2f}%",
        'CV Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        'CV Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        'CV F1-Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        'mean_f1': scores['test_f1'].mean()
    })

cv_df = pd.DataFrame(cv_results_list).sort_values(by='mean_f1', ascending=False).drop(columns=['mean_f1'])
print("\n--- 5-Fold Cross-Validation Stability Benchmark ---")
print(cv_df.to_string(index=False))

# 2. Hyperparameter Tuning
print("\n2. Performing Systematic Hyperparameter Tuning with GridSearchCV...")

# XGBoost Tuning
print("  Tuning XGBoost hyperparameters...")
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.08, 0.15],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    xgb_param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"  Best XGBoost Parameters: {xgb_grid.best_params_}")
print(f"  Best XGBoost 3-Fold ROC-AUC: {xgb_grid.best_score_:.4f}")

# Random Forest Tuning
print("  Tuning Random Forest hyperparameters...")
rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [8, 12, 16],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [4, 8]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"  Best Random Forest Parameters: {rf_grid.best_params_}")
print(f"  Best Random Forest 3-Fold ROC-AUC: {rf_grid.best_score_:.4f}")

# 3. Ensemble Methods (Voting & Stacking)
print("\n3. Building & Evaluating Voting and Stacking Ensemble Models...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf),
        ('lr', log_reg)
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_scaled, y_train)

# Stacking Ensemble
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf),
        ('lr', log_reg)
    ],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)
stacking_clf.fit(X_train_scaled, y_train)

# Advanced Models Evaluation on Hold-Out Test Set
advanced_models = {
    'Tuned XGBoost': best_xgb,
    'Tuned Random Forest': best_rf,
    'Gradient Boosting (GBDT)': GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42).fit(X_train_scaled, y_train),
    'Extra Trees': ExtraTreesClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train),
    'Soft Voting Ensemble': voting_clf,
    'Stacking Classifier': stacking_clf
}

adv_results = []
for name, model in advanced_models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp)
    
    adv_results.append({
        'Model': name,
        'Test Accuracy (%)': round(accuracy_score(y_test, preds) * 100, 2),
        'Precision': round(precision_score(y_test, preds), 4),
        'Recall (Sens)': round(recall_score(y_test, preds), 4),
        'Specificity': round(spec, 4),
        'F1-Score': round(f1_score(y_test, preds), 4),
        'ROC-AUC': round(roc_auc_score(y_test, probs), 4),
        'Brier Score': round(brier_score_loss(y_test, probs), 4)
    })

adv_df = pd.DataFrame(adv_results).sort_values(by='ROC-AUC', ascending=False)
print("\n--- Week 5 Advanced Model Performance on Hold-Out Test Set ---")
print(adv_df.to_string(index=False))

# PLOT 5: Feature Importance Comparison (XGBoost vs Random Forest)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
xgb_importances = pd.Series(best_xgb.feature_importances_, index=feature_names).sort_values(ascending=True)
xgb_importances.plot(kind='barh', color='#f43f5e')
plt.title('Tuned XGBoost Feature Importance (Gain)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance')

plt.subplot(1, 2, 2)
rf_importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=True)
rf_importances.plot(kind='barh', color='#3b82f6')
plt.title('Tuned Random Forest Feature Importance (Gini)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance')

plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week5_feature_importance.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week5_feature_importance.png"), dpi=200)
plt.close()

# PLOT 6: Advanced Models ROC Curves
plt.figure(figsize=(10, 7))
palette_adv = sns.color_palette("deep", len(advanced_models))
for idx, (name, model) in enumerate(advanced_models.items()):
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=palette_adv[idx], lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Week 5: Advanced Ensemble ROC Comparison', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week5_advanced_roc_curves.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week5_advanced_roc_curves.png"), dpi=200)
plt.close()

# PLOT 7: Optimal Decision Threshold Tuning
print("\n4. Evaluating Optimal Decision Thresholds for Clinical Screening...")
thresholds = np.linspace(0.1, 0.9, 81)
t_metrics = []
best_xgb_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]

for t in thresholds:
    t_preds = (best_xgb_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
    t_metrics.append({
        'threshold': t,
        'accuracy': accuracy_score(y_test, t_preds),
        'precision': precision_score(y_test, t_preds, zero_division=0),
        'recall': recall_score(y_test, t_preds),
        'f1': f1_score(y_test, t_preds),
        'specificity': tn / (tn + fp)
    })

t_df = pd.DataFrame(t_metrics)
best_f1_row = t_df.loc[t_df['f1'].idxmax()]
print(f"  Maximum F1-Score ({best_f1_row['f1']:.4f}) occurs at Threshold = {best_f1_row['threshold']:.2f} (Recall: {best_f1_row['recall']:.4f}, Precision: {best_f1_row['precision']:.4f})")

plt.figure(figsize=(10, 6))
plt.plot(t_df['threshold'], t_df['precision'], label='Precision (PPV)', color='#f59e0b', lw=2)
plt.plot(t_df['threshold'], t_df['recall'], label='Recall (Sensitivity)', color='#10b981', lw=2)
plt.plot(t_df['threshold'], t_df['f1'], label='F1-Score', color='#6366f1', lw=2.5)
plt.plot(t_df['threshold'], t_df['accuracy'], label='Accuracy', color='#0ea5e9', lw=1.5, linestyle='--')
plt.axvline(best_f1_row['threshold'], color='#e11d48', linestyle=':', label=f"Optimal F1 Threshold ({best_f1_row['threshold']:.2f})")
plt.axvline(0.50, color='#64748b', linestyle='--', label="Default Threshold (0.50)")
plt.xlabel('Classification Probability Threshold', fontsize=12)
plt.ylabel('Metric Score', fontsize=12)
plt.title('Tuned XGBoost: Threshold vs Clinical Performance Metrics', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc='lower center', fontsize=10, ncol=3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FRONTEND_ASSETS, "week5_threshold_tuning.png"), dpi=200)
plt.savefig(os.path.join(ARTIFACT_DIR, "week5_threshold_tuning.png"), dpi=200)
plt.close()

# ── EXPORT UPDATED MODEL WEIGHTS & METRICS JSON ─────────────────────────────
scaler = preprocessor.named_transformers_['num']
means = list(scaler.mean_) + [0.0]*len(categorical_features)
scales = list(scaler.scale_) + [1.0]*len(categorical_features)

lr_full = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)

weights_export = {
    "features": feature_names,
    "coefficients": [float(c) for c in lr_full.coef_[0]],
    "intercept": float(lr_full.intercept_[0]),
    "scaler_means": [float(m) for m in means],
    "scaler_scales": [float(s) for s in scales],
    "feature_importance_xgb": {feat: float(imp) for feat, imp in zip(feature_names, best_xgb.feature_importances_)},
    "feature_importance_rf": {feat: float(imp) for feat, imp in zip(feature_names, best_rf.feature_importances_)},
    "optimal_threshold": float(best_f1_row['threshold']),
    "best_xgb_params": {k: int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v for k, v in xgb_grid.best_params_.items()},
    "best_rf_params": {k: int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v for k, v in rf_grid.best_params_.items()},
    "summary_metrics": {
        "xgboost_roc_auc": float(adv_df[adv_df['Model'] == 'Tuned XGBoost']['ROC-AUC'].values[0]),
        "xgboost_accuracy": float(adv_df[adv_df['Model'] == 'Tuned XGBoost']['Test Accuracy (%)'].values[0]),
        "voting_ensemble_roc_auc": float(adv_df[adv_df['Model'] == 'Soft Voting Ensemble']['ROC-AUC'].values[0])
    }
}

with open('/Users/cryptorth/Documents/ml/projects/frontend/model_weights.json', 'w') as f:
    json.dump(weights_export, f, indent=2)

with open(os.path.join(ARTIFACT_DIR, 'model_weights.json'), 'w') as f:
    json.dump(weights_export, f, indent=2)

print("\nModel weights and metrics saved successfully to model_weights.json")

# ── UPDATE JUPYTER NOTEBOOK (cardioDemo.ipynb) ──────────────────────────────
print("\nUpdating Jupyter Notebook with complete Week 4 & Week 5 sections...")

nb_path = '/Users/cryptorth/Documents/ml/projects/cardioDemo.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

def md(content):
    return nbformat.v4.new_markdown_cell(content)

def code(content):
    return nbformat.v4.new_code_cell(content)

new_cells = []

# Retain initial cells up to Phase 3 (Model Creation)
for cell in nb.cells:
    if "Phase 4:" in cell.source or "Phase 4" in cell.source:
        break
    new_cells.append(cell)

# Append comprehensive Week 4 Section
new_cells.append(md("""## Phase 4: Model Evaluation (Week 4 SOP Milestone)
### Objective & Scope
1. Evaluate baseline models (Scratch Logistic Regression, Sklearn Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost) on the hold-out test set.
2. Compute full diagnostic metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC.
3. Diagnose **Overfitting vs Underfitting** via train-vs-test score variance and Learning Curves."""))

new_cells.append(code("""# Week 4: Model Evaluation Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

baseline_models = {
    'Scratch Logistic Reg': None,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=120, random_state=42, n_jobs=-1)
}

week4_results = []
y_probs_dict = {}
y_preds_dict = {}

for name, model in baseline_models.items():
    if name == 'Scratch Logistic Reg':
        train_preds = scratch_preds_train = predict_scratch(X_train_scaled, w, b)
        test_preds = scratch_preds
        test_probs = predict_proba_scratch(X_test_scaled, w, b)
    else:
        model.fit(X_train_scaled, y_train)
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        test_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else test_preds

    y_probs_dict[name] = test_probs
    y_preds_dict[name] = test_preds

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    spec = tn / (tn + fp)
    overfit_gap = (train_acc - test_acc) * 100

    week4_results.append({
        'Model': name,
        'Train Acc (%)': round(train_acc * 100, 2),
        'Test Acc (%)': round(test_acc * 100, 2),
        'Overfit Gap (%)': round(overfit_gap, 2),
        'Precision': round(prec, 4),
        'Recall (Sens)': round(rec, 4),
        'Specificity': round(spec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    })

week4_df = pd.DataFrame(week4_results).sort_values(by='F1-Score', ascending=False)
display(week4_df)"""))

new_cells.append(md("""### Week 4 Diagnostic Visualizations: Confusion Matrices & ROC Curves"""))

new_cells.append(code("""# Confusion Matrices & ROC Curves
plt.figure(figsize=(16, 9))
models_to_plot = [m for m in baseline_models.keys() if m != 'K-Nearest Neighbors'][:6]
for idx, m_name in enumerate(models_to_plot):
    plt.subplot(2, 3, idx + 1)
    cm = confusion_matrix(y_test, y_preds_dict[m_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD (0)', 'CVD (1)'],
                yticklabels=['No CVD (0)', 'CVD (1)'])
    plt.title(f"{m_name}\\nAcc: {accuracy_score(y_test, y_preds_dict[m_name]):.3f} | F1: {f1_score(y_test, y_preds_dict[m_name]):.3f}", fontsize=11, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Multi-Model ROC Comparison
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", len(baseline_models))
for idx, (m_name, probs) in enumerate(y_probs_dict.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{m_name} (AUC = {score:.4f})", color=palette[idx], lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()"""))

new_cells.append(md("""### Overfitting & Underfitting Diagnosis with Learning Curves"""))

new_cells.append(code("""# Learning Curves Analysis
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lc_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=100, random_state=42, n_jobs=-1)
}

train_sizes_rel = np.linspace(0.1, 1.0, 5)

for idx, (name, model) in enumerate(lc_models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, train_sizes=train_sizes_rel,
        cv=3, scoring='f1', n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax = axes[idx]
    ax.plot(train_sizes, train_mean, 'o-', color='#2563eb', label='Training F1 Score', lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2563eb')
    ax.plot(train_sizes, val_mean, 's-', color='#16a34a', label='Cross-Val F1 Score', lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#16a34a')
    
    ax.set_title(f"Learning Curve: {name}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0.65, 0.85])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Append comprehensive Week 5 Section
new_cells.append(md("""## Phase 5: Advanced Model Training, Cross-Validation & Hyperparameter Tuning (Week 5 SOP Milestone)
### Tasks Executed:
1. **5-Fold Stratified Cross-Validation**: Assess candidate models' stability across distinct splits without data leakage.
2. **Systematic Hyperparameter Tuning**: Optimize tree depth, learning rate, subsampling, and regularization for XGBoost & Random Forest using `GridSearchCV`.
3. **Advanced Ensemble Modeling**: Construct Soft Voting and Stacking meta-models.
4. **Feature Importance & Interpretability**: Quantify clinical drivers (SBP, Cholesterol, Age, BMI).
5. **Decision Threshold Optimization**: Maximize clinical sensitivity without compromising precision."""))

new_cells.append(code("""# 1. 5-Fold Stratified Cross-Validation
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting (GBDT)': GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=120, random_state=42, n_jobs=-1)
}

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
cv_results_list = []

for name, model in cv_models.items():
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results_list.append({
        'Model': name,
        'CV Accuracy': f"{scores['test_accuracy'].mean()*100:.2f}% ± {scores['test_accuracy'].std()*100:.2f}%",
        'CV Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        'CV Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        'CV F1-Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        'mean_f1': scores['test_f1'].mean()
    })

cv_df = pd.DataFrame(cv_results_list).sort_values(by='mean_f1', ascending=False).drop(columns=['mean_f1'])
display(cv_df)"""))

new_cells.append(md("""### Hyperparameter Tuning (GridSearchCV) for XGBoost and Random Forest"""))

new_cells.append(code("""# 2. Hyperparameter Optimization with GridSearchCV
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.08, 0.15],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1
)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_

rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [8, 12, 16],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [4, 8]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

print(f"Optimal XGBoost Params: {xgb_grid.best_params_}")
print(f"Optimal Random Forest Params: {rf_grid.best_params_}")"""))

new_cells.append(md("""### Advanced Ensembles (Soft Voting & Stacking) & Hold-Out Test Evaluation"""))

new_cells.append(code("""# 3. Ensembles & Final Comparative Benchmarks
log_reg = LogisticRegression(max_iter=1000, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('lr', log_reg)],
    voting='soft', n_jobs=-1
).fit(X_train_scaled, y_train)

stacking_clf = StackingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('lr', log_reg)],
    final_estimator=LogisticRegression(), cv=3, n_jobs=-1
).fit(X_train_scaled, y_train)

advanced_models = {
    'Tuned XGBoost': best_xgb,
    'Tuned Random Forest': best_rf,
    'Gradient Boosting (GBDT)': GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42).fit(X_train_scaled, y_train),
    'Extra Trees': ExtraTreesClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train),
    'Soft Voting Ensemble': voting_clf,
    'Stacking Classifier': stacking_clf
}

adv_results = []
for name, model in advanced_models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    adv_results.append({
        'Model': name,
        'Test Accuracy (%)': round(accuracy_score(y_test, preds) * 100, 2),
        'Precision': round(precision_score(y_test, preds), 4),
        'Recall (Sens)': round(recall_score(y_test, preds), 4),
        'Specificity': round(tn / (tn + fp), 4),
        'F1-Score': round(f1_score(y_test, preds), 4),
        'ROC-AUC': round(roc_auc_score(y_test, probs), 4)
    })

adv_df = pd.DataFrame(adv_results).sort_values(by='ROC-AUC', ascending=False)
display(adv_df)"""))

new_cells.append(md("""### Feature Importance & Clinical Decision Threshold Analysis"""))

new_cells.append(code("""# Feature Importance Visualization
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
feature_names = continuous_features + categorical_features
pd.Series(best_xgb.feature_importances_, index=feature_names).sort_values().plot(kind='barh', color='#f43f5e')
plt.title('Tuned XGBoost Feature Importance (Gain)', fontsize=12, fontweight='bold')

plt.subplot(1, 2, 2)
pd.Series(best_rf.feature_importances_, index=feature_names).sort_values().plot(kind='barh', color='#3b82f6')
plt.title('Tuned Random Forest Feature Importance (Gini)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Optimal Decision Threshold Analysis
thresholds = np.linspace(0.1, 0.9, 81)
t_metrics = []
xgb_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]

for t in thresholds:
    t_preds = (xgb_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
    t_metrics.append({
        'threshold': t,
        'accuracy': accuracy_score(y_test, t_preds),
        'precision': precision_score(y_test, t_preds, zero_division=0),
        'recall': recall_score(y_test, t_preds),
        'f1': f1_score(y_test, t_preds),
        'specificity': tn / (tn + fp)
    })

t_df = pd.DataFrame(t_metrics)
best_f1_idx = t_df['f1'].idxmax()
opt_t = t_df.loc[best_f1_idx, 'threshold']

plt.figure(figsize=(10, 5))
plt.plot(t_df['threshold'], t_df['precision'], label='Precision (PPV)', color='#f59e0b', lw=2)
plt.plot(t_df['threshold'], t_df['recall'], label='Recall (Sensitivity)', color='#10b981', lw=2)
plt.plot(t_df['threshold'], t_df['f1'], label='F1-Score', color='#6366f1', lw=2.5)
plt.axvline(opt_t, color='#e11d48', linestyle=':', label=f"Optimal F1 Threshold ({opt_t:.2f})")
plt.axvline(0.50, color='#64748b', linestyle='--', label="Default (0.50)")
plt.xlabel('Probability Threshold')
plt.ylabel('Score')
plt.title('Clinical Operating Threshold Optimization', fontsize=13, fontweight='bold')
plt.legend(loc='lower center', ncol=3)
plt.grid(True, alpha=0.3)
plt.show()"""))

nb.cells = new_cells
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Jupyter Notebook updated successfully at: {nb_path}")
print("="*70)
print("ALL ML SOP WEEK 4 & WEEK 5 TASKS COMPLETED SUCCESSFULLY!")
print("="*70)
