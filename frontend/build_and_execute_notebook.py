import os
import sys
import json
import nbformat
from nbclient import NotebookClient

# Ensure OpenMP runtime for macOS
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

nb_path = "/Users/cryptorth/Documents/ml/projects/cardioDemo.ipynb"

def md_cell(source):
    return nbformat.v4.new_markdown_cell(source)

def code_cell(source):
    return nbformat.v4.new_code_cell(source)

cells = []

# Title & Metadata
cells.append(md_cell("""# Cardiovascular Disease Prediction using Machine Learning
**Standard Operating Procedure (SOP) Project Milestone**  
Department of Computer Engineering | Academic Year 2025–2026"""))

cells.append(md_cell("""## Problem Statement & Clinical Objective
Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide. Early detection and proactive lifestyle intervention are essential for reducing cardiovascular mortality. This project develops an end-to-end Machine Learning pipeline combining data pre-processing, clinical feature engineering, scratch algorithmic implementation (mandatory SOP requirement), comprehensive model evaluation, cross-validation, and systematic hyperparameter tuning."""))

cells.append(md_cell("""## Dataset Information
- **Dataset Source:** Kaggle (sulianova/cardiovascular-disease-dataset)
- **Rows:** 70,000 patient examination records
- **Features:** 11 clinical features (Age, Gender, Height, Weight, SBP, DBP, Cholesterol, Glucose, Smoking, Alcohol, Physical Activity)
- **Target Variable:** `cardio` (0: Absence of CVD, 1: Presence of CVD)"""))

# Cell: Environment & Core Imports
cells.append(code_cell("""import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set library path for macOS OpenMP runtime
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Core Data & Numerical Computing
import numpy as np
import pandas as pd
import scipy as sp

# Data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Scikit-Learn Preprocessing & Splitting
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    learning_curve, GridSearchCV, RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Scikit-Learn Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, auc
)

# Baseline & Advanced Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

print("All libraries imported successfully!")"""))

# Cell: Load Dataset
cells.append(md_cell("""## Phase 1: Data Exploration & Initial Inspection"""))
cells.append(code_cell("""# Load the dataset dynamically across working directories
csv_candidates = [
    "cardio/cardio_train.csv",
    "../cardio/cardio_train.csv",
    "/Users/cryptorth/Documents/ml/projects/cardio/cardio_train.csv",
    os.path.join(os.getcwd(), 'cardio', 'cardio_train.csv'),
    os.path.join(os.getcwd(), '..', 'cardio', 'cardio_train.csv')
]

data_path = None
for p in csv_candidates:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("Could not locate cardio_train.csv")

df = pd.read_csv(data_path, sep=";")
print(f"Dataset Loaded Successfully from '{data_path}'! Shape: {df.shape}")
df.head()"""))

cells.append(md_cell("""### Age Transformation & Structural Inspection
Converting patient age from days to years."""))
cells.append(code_cell("""# Convert age from days to integer years
df['age'] = (df['age'] / 365.25).round().astype(int)
df.head()"""))

cells.append(code_cell("""df.info()"""))
cells.append(code_cell("""df.describe()"""))
cells.append(code_cell("""print("Missing Values Check:")\nprint(df.isnull().sum())"""))

# Cell: EDA
cells.append(md_cell("""## Exploratory Data Analysis (EDA)
### Target Class Distribution"""))
cells.append(code_cell("""plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=df, palette=['#3b82f6', '#ef4444'])
plt.title('Target Distribution: Cardio (0: Absence, 1: Presence)', fontsize=12, fontweight='bold')
plt.xlabel('Cardiovascular Disease Status')
plt.ylabel('Patient Count')
plt.show()"""))

cells.append(md_cell("""### Continuous & Categorical Feature Distributions"""))
cells.append(code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
for i, feature in enumerate(features):
    sns.histplot(df[feature], kde=True, ax=axes[i//3, i%3], color='#0284c7', bins=30)
    axes[i//3, i%3].set_title(f'Distribution of {feature}', fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
cat_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
for i, feature in enumerate(cat_features):
    sns.countplot(x=feature, hue='cardio', data=df, ax=axes[i//3, i%3], palette=['#60a5fa', '#f87171'])
    axes[i//3, i%3].set_title(f'{feature} vs Cardio Target', fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(md_cell("""### Feature Correlation Analysis"""))
cells.append(code_cell("""plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.show()"""))

# Phase 2: Data Cleaning & Preprocessing
cells.append(md_cell("""## Phase 2: Data Cleaning, Outlier Removal & Feature Engineering
Removing duplicate records and physiologically impossible outliers according to medical standards:
- Systolic Blood Pressure (`ap_hi`): $[60, 240]$ mmHg
- Diastolic Blood Pressure (`ap_lo`): $[40, 140]$ mmHg with $ap\_lo < ap\_hi$
- Height: $[100, 220]$ cm, Weight: $[30, 200]$ kg
- Engineering **Body Mass Index (BMI)** and **Pulse Pressure**."""))

cells.append(code_cell("""# 1. Deduplication
initial_len = len(df)
df.drop_duplicates(inplace=True)

# 2. Outlier filtering
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[df['ap_lo'] < df['ap_hi']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

# 3. Clinical Feature Engineering
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

print(f"Original samples: {initial_len}")
print(f"Cleaned samples: {len(df)} (Removed {initial_len - len(df)} anomalies)")
df.head()"""))

cells.append(md_cell("""### Feature Scaling & Stratified Train/Test Split"""))
cells.append(code_cell("""X = df.drop(['id', 'cardio'], axis=1)
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

print(f"Training feature matrix: {X_train_scaled.shape}")
print(f"Testing feature matrix:  {X_test_scaled.shape}")"""))

# Phase 3: Scratch Implementation (Mandatory SOP Requirement)
cells.append(md_cell("""## Phase 3: Model Creation — Scratch Logistic Regression (SOP Mandatory Constraint)
Implementing Logistic Regression and Gradient Descent from mathematical first principles without external ML libraries."""))

cells.append(code_cell("""# Mathematical scratch implementation of Logistic Regression
def logistic_regression_scratch(X, y, lr=0.1, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0

    for _ in range(iterations):
        linear = np.dot(X, weights) + bias
        # Sigmoid activation with numerical clipping
        y_hat = 1 / (1 + np.exp(-np.clip(linear, -250, 250)))
        
        # Gradient computation
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        
        # Parameter update
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

def predict_proba_scratch(X, weights, bias):
    linear = np.dot(X, weights) + bias
    return 1 / (1 + np.exp(-np.clip(linear, -250, 250)))

def predict_scratch(X, weights, bias, threshold=0.5):
    probs = predict_proba_scratch(X, weights, bias)
    return (probs >= threshold).astype(int)

print("Training Scratch Logistic Regression (1000 iterations, lr=0.1)...")
w_scratch, b_scratch = logistic_regression_scratch(X_train_scaled, y_train.to_numpy(), lr=0.1, iterations=1000)

scratch_train_preds = predict_scratch(X_train_scaled, w_scratch, b_scratch)
scratch_test_preds = predict_scratch(X_test_scaled, w_scratch, b_scratch)
scratch_test_probs = predict_proba_scratch(X_test_scaled, w_scratch, b_scratch)

print(f"Scratch LR Test Accuracy:  {accuracy_score(y_test, scratch_test_preds):.4f}")
print(f"Scratch LR Test Precision: {precision_score(y_test, scratch_test_preds):.4f}")
print(f"Scratch LR Test Recall:    {recall_score(y_test, scratch_test_preds):.4f}")
print(f"Scratch LR Test F1-Score:  {f1_score(y_test, scratch_test_preds):.4f}")
print(f"Scratch LR Test ROC-AUC:   {roc_auc_score(y_test, scratch_test_probs):.4f}")"""))

# Phase 4: Week 4 Model Evaluation
cells.append(md_cell("""## Phase 4: Model Evaluation (Week 4 SOP Milestone)
### Objective
1. Evaluate baseline models (Scratch LR, Sklearn LR, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost) on hold-out test set.
2. Compute full diagnostic metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, and ROC-AUC.
3. Diagnose **Overfitting vs Underfitting** via training vs testing score variance and Learning Curves."""))

cells.append(code_cell("""# Week 4: Model Evaluation Benchmark
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
        train_preds = scratch_train_preds
        test_preds = scratch_test_preds
        test_probs = scratch_test_probs
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

cells.append(md_cell("""### Week 4 Diagnostic Plots: Confusion Matrices & ROC Curves"""))
cells.append(code_cell("""# 1. Confusion Matrices
plt.figure(figsize=(16, 9))
models_to_plot = [m for m in baseline_models.keys() if m != 'K-Nearest Neighbors'][:6]
for idx, m_name in enumerate(models_to_plot):
    plt.subplot(2, 3, idx + 1)
    cm = confusion_matrix(y_test, y_preds_dict[m_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD (0)', 'CVD (1)'],
                yticklabels=['No CVD (0)', 'CVD (1)'])
    plt.title(f"{m_name}\\nAcc: {accuracy_score(y_test, y_preds_dict[m_name]):.3f} | F1: {f1_score(y_test, y_preds_dict[m_name]):.3f}", fontsize=11, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 2. Multi-Model ROC Comparison
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

cells.append(md_cell("""### Overfitting & Underfitting Diagnosis with Learning Curves"""))
cells.append(code_cell("""# 3. Learning Curves Analysis
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

# Phase 5: Week 5 Advanced Training
cells.append(md_cell("""## Phase 5: Advanced Model Training, Cross-Validation & Hyperparameter Tuning (Week 5 SOP Milestone)
### Tasks Executed:
1. **5-Fold Stratified Cross-Validation**: Assess candidate models' stability across distinct folds.
2. **Systematic Hyperparameter Tuning**: Optimize tree depth, learning rate, subsampling, and regularization for XGBoost & Random Forest using `GridSearchCV`.
3. **Advanced Ensemble Modeling**: Construct Soft Voting and Stacking meta-models.
4. **Feature Importance & Interpretability**: Quantify clinical drivers (SBP, Cholesterol, Age, BMI).
5. **Decision Threshold Optimization**: Maximize clinical sensitivity without compromising precision."""))

cells.append(code_cell("""# 1. 5-Fold Stratified Cross-Validation
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

cells.append(md_cell("""### Hyperparameter Tuning (GridSearchCV) for XGBoost and Random Forest"""))
cells.append(code_cell("""# 2. Hyperparameter Optimization with GridSearchCV
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

cells.append(md_cell("""### Advanced Ensembles (Soft Voting & Stacking) & Hold-Out Test Evaluation"""))
cells.append(code_cell("""# 3. Ensembles & Final Comparative Benchmarks
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

cells.append(md_cell("""### Feature Importance & Clinical Decision Threshold Analysis"""))
cells.append(code_cell("""# Feature Importance Visualization
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

# Construct notebook and write
nb = nbformat.v4.new_notebook(cells=cells)
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook written with {len(cells)} cells. Now executing end-to-end to verify and save execution outputs...")

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    resources={'metadata': {'path': '/Users/cryptorth/Documents/ml/projects'}}
)
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("SUCCESS: Notebook executed completely with 0 errors and all outputs saved!")
