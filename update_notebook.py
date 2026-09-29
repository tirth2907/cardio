import os
import sys
import json
import nbformat
from nbclient import NotebookClient

# Ensure OpenMP path
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

nb_path = "/Users/cryptorth/Documents/ml/projects/cardioDemo.ipynb"

def md(content):
    return nbformat.v4.new_markdown_cell(content.strip())

def code(content):
    return nbformat.v4.new_code_cell(content.strip())

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md("""
# Cardiovascular Disease Prediction — Modular ML Pipeline
**Standard Operating Procedure (SOP) Project Milestone**  
Department of Computer Engineering | Academic Year 2025–2026
"""))

# ── Section 1: Environment & Setup ──────────────────────────────────────────
cells.append(md("""
## 1. Environment Configuration & Library Imports
In this section, we import all necessary numerical, visualization, machine learning, and evaluation libraries.
"""))

cells.append(md("### 1.1 System & Environment Path Setup"))
cells.append(code("""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set library path for macOS OpenMP runtime (for XGBoost acceleration)
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
"""))

cells.append(md("### 1.2 Core Data Manipulation Libraries"))
cells.append(code("""
import numpy as np
import pandas as pd
import scipy as sp
"""))

cells.append(md("### 1.3 Visualization Libraries & Theming"))
cells.append(code("""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
"""))

cells.append(md("### 1.4 Scikit-Learn Preprocessing & Splitting Modules"))
cells.append(code("""
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
"""))

cells.append(md("### 1.5 Evaluation Metrics Modules"))
cells.append(code("""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    auc
)
"""))

cells.append(md("### 1.6 Machine Learning Classifiers & Ensemble Models"))
cells.append(code("""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib

print("✅ All libraries and modules successfully loaded!")
"""))

# ── Section 2: Data Loading & Initial Inspection ─────────────────────────────
cells.append(md("""
## 2. Dataset Loading & Exploration (Phase 1)
Loading the 70,000 patient cardiovascular disease records and performing structural inspection.
"""))

cells.append(md("### 2.1 Load Dataset with Dynamic Path Resolution"))
cells.append(code("""
# Candidate file paths for robust multi-environment loading
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
print(f"Dataset Loaded Successfully from '{data_path}'!")
print(f"Initial Shape: {df.shape[0]:,} rows and {df.shape[1]} columns")
"""))

cells.append(md("### 2.2 First 5 Records Preview"))
cells.append(code("""
df.head()
"""))

cells.append(md("### 2.3 Convert Patient Age from Days to Years"))
cells.append(code("""
# Transform age (days -> integer years)
df['age'] = (df['age'] / 365.25).round().astype(int)
df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio']].head()
"""))

cells.append(md("### 2.4 Dataset Info & Data Types"))
cells.append(code("""
df.info()
"""))

cells.append(md("### 2.5 Summary Statistics"))
cells.append(code("""
df.describe().T
"""))

cells.append(md("### 2.6 Missing Value Analysis"))
cells.append(code("""
missing_vals = df.isnull().sum()
print("Missing values count per feature:")
print(missing_vals)
"""))

# ── Section 3: EDA ───────────────────────────────────────────────────────────
cells.append(md("""
## 3. Exploratory Data Analysis (EDA)
Visualizing class balance, continuous distributions, categorical correlations, and multi-collinearity.
"""))

cells.append(md("### 3.1 Target Variable Distribution (`cardio`)"))
cells.append(code("""
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='cardio', data=df, palette=['#38bdf8', '#f87171'])
plt.title('Distribution of Target Variable (0: Healthy, 1: CVD)', fontsize=12, fontweight='bold')
plt.xlabel('Cardiovascular Disease')
plt.ylabel('Patient Count')

# Annotate counts
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.2 Continuous Features Distribution"))
cells.append(code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

for idx, col in enumerate(continuous_cols):
    ax = axes[idx // 3, idx % 3]
    sns.histplot(df[col], kde=True, ax=ax, color='#0284c7', bins=30)
    ax.set_title(f'Distribution of {col}', fontweight='bold')

axes[1, 2].set_visible(False) # Hide extra subplot
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.3 Categorical Features vs CVD Target"))
cells.append(code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

for idx, col in enumerate(cat_cols):
    ax = axes[idx // 3, idx % 3]
    sns.countplot(x=col, hue='cardio', data=df, ax=ax, palette=['#60a5fa', '#f87171'])
    ax.set_title(f'{col} vs Cardio Status', fontweight='bold')
    ax.legend(['No CVD', 'CVD'])

plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.4 Feature Correlation Heatmap"))
cells.append(code("""
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.show()
"""))

cells.append(md("### 3.5 Clinical Age & Weight Distribution Boxplots"))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='cardio', y='age', data=df, ax=axes[0], palette=['#93c5fd', '#fca5a5'])
axes[0].set_title('Age vs Cardiovascular Disease', fontweight='bold')
axes[0].set_xticklabels(['No CVD (0)', 'CVD (1)'])

sns.boxplot(x='cardio', y='weight', data=df, ax=axes[1], palette=['#93c5fd', '#fca5a5'])
axes[1].set_title('Weight vs Cardiovascular Disease', fontweight='bold')
axes[1].set_xticklabels(['No CVD (0)', 'CVD (1)'])

plt.tight_layout()
plt.show()
"""))

# ── Section 4: Cleaning & Feature Engineering ────────────────────────────────
cells.append(md("""
## 4. Data Pre-processing & Feature Engineering (Phase 2)
Filtering physiological anomalies and computing clinical risk indicators (**BMI** and **Pulse Pressure**).
"""))

cells.append(md("### 4.1 Remove Duplicate Records"))
cells.append(code("""
initial_len = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_len - len(df)}")
"""))

cells.append(md("### 4.2 Physiological Outlier Removal"))
cells.append(code("""
# Physiological thresholds based on medical guidelines
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[df['ap_lo'] < df['ap_hi']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

print(f"Cleaned dataset samples: {len(df):,} (Filtered {initial_len - len(df):,} invalid rows)")
"""))

cells.append(md("### 4.3 Feature Engineering: Body Mass Index (BMI)"))
cells.append(code("""
# BMI = weight (kg) / (height (m))^2
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
print("BMI summary:")
print(df['bmi'].describe())
"""))

cells.append(md("### 4.4 Feature Engineering: Pulse Pressure (PP)"))
cells.append(code("""
# Pulse Pressure = Systolic BP - Diastolic BP
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
print("Pulse Pressure summary:")
print(df['pulse_pressure'].describe())
"""))

cells.append(md("### 4.5 Train / Test Stratified Split (80% Train, 20% Test)"))
cells.append(code("""
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Testing set:  {X_test.shape[0]:,} samples")
"""))

cells.append(md("### 4.6 Feature Standardization (ColumnTransformer)"))
cells.append(code("""
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

print("StandardScaler applied to continuous features successfully!")
"""))

# ── Section 5: Scratch Implementation ────────────────────────────────────────
cells.append(md("""
## 5. Scratch Implementation (SOP Mandatory Requirement - Phase 3)
Implementing Logistic Regression and Gradient Descent from scratch without using scikit-learn.
"""))

cells.append(md("### 5.1 Scratch Logistic Regression Model Definition"))
cells.append(code("""
def logistic_regression_scratch(X, y, lr=0.1, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0

    for i in range(iterations):
        linear = np.dot(X, weights) + bias
        # Sigmoid with numerical clipping to prevent overflow
        y_hat = 1 / (1 + np.exp(-np.clip(linear, -250, 250)))
        
        # Gradients
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        
        # Gradient descent update
        weights -= lr * dw
        bias -= lr * db

    return weights, bias
"""))

cells.append(md("### 5.2 Scratch Prediction & Probability Functions"))
cells.append(code("""
def predict_proba_scratch(X, weights, bias):
    linear = np.dot(X, weights) + bias
    return 1 / (1 + np.exp(-np.clip(linear, -250, 250)))

def predict_scratch(X, weights, bias, threshold=0.5):
    probs = predict_proba_scratch(X, weights, bias)
    return (probs >= threshold).astype(int)
"""))

cells.append(md("### 5.3 Train Scratch Logistic Regression"))
cells.append(code("""
print("Training Scratch Logistic Regression (1000 iterations, learning_rate=0.1)...")
w_scratch, b_scratch = logistic_regression_scratch(
    X_train_scaled, y_train.to_numpy(), lr=0.1, iterations=1000
)

scratch_train_preds = predict_scratch(X_train_scaled, w_scratch, b_scratch)
scratch_test_preds = predict_scratch(X_test_scaled, w_scratch, b_scratch)
scratch_test_probs = predict_proba_scratch(X_test_scaled, w_scratch, b_scratch)
print("✅ Scratch Model Training Complete!")
"""))

cells.append(md("### 5.4 Evaluate Scratch Model Test Performance"))
cells.append(code("""
print("=== Scratch Logistic Regression Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, scratch_test_preds):.4f}")
print(f"Precision: {precision_score(y_test, scratch_test_preds):.4f}")
print(f"Recall:    {recall_score(y_test, scratch_test_preds):.4f}")
print(f"F1-Score:  {f1_score(y_test, scratch_test_preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, scratch_test_probs):.4f}")
"""))

# ── Section 6: Week 4 Model Evaluation ───────────────────────────────────────
cells.append(md("""
## 6. Baseline Models Evaluation (Week 4 SOP Milestone)
Comparing the scratch implementation against library-based models on the hold-out test set.
"""))

cells.append(md("### 6.1 Initialize Baseline Models Dictionary"))
cells.append(code("""
baseline_models = {
    'Scratch Logistic Reg': None,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=120, random_state=42, n_jobs=-1)
}
"""))

cells.append(md("### 6.2 Train & Evaluate Baseline Models"))
cells.append(code("""
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
"""))

cells.append(md("### 6.3 Baseline Performance Benchmark Table"))
cells.append(code("""
week4_df = pd.DataFrame(week4_results).sort_values(by='F1-Score', ascending=False)
display(week4_df)
"""))

cells.append(md("### 6.4 Confusion Matrices of Baseline Models"))
cells.append(code("""
plt.figure(figsize=(16, 9))
models_to_plot = [m for m in baseline_models.keys() if m != 'K-Nearest Neighbors'][:6]

for idx, m_name in enumerate(models_to_plot):
    plt.subplot(2, 3, idx + 1)
    cm = confusion_matrix(y_test, y_preds_dict[m_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD (0)', 'CVD (1)'],
                yticklabels=['No CVD (0)', 'CVD (1)'])
    plt.title(f"{m_name}\\nAcc: {accuracy_score(y_test, y_preds_dict[m_name]):.3f} | F1: {f1_score(y_test, y_preds_dict[m_name]):.3f}",
              fontsize=11, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()
"""))

cells.append(md("### 6.5 ROC Curves Comparison"))
cells.append(code("""
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
plt.show()
"""))

cells.append(md("### 6.6 Precision-Recall Curves Comparison"))
cells.append(code("""
plt.figure(figsize=(10, 6))

for idx, (m_name, probs) in enumerate(y_probs_dict.items()):
    pr, rc, _ = precision_recall_curve(y_test, probs)
    ap = auc(rc, pr)
    plt.plot(rc, pr, label=f"{m_name} (AP = {ap:.4f})", color=palette[idx], lw=2)

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve Comparison', fontsize=13, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()
"""))

# ── Section 7: Overfitting & Underfitting Diagnosis ──────────────────────────
cells.append(md("""
## 7. Overfitting & Underfitting Diagnosis (Week 4 Milestone)
Analyzing learning curves across sample sizes ($10\%$ to $100\%$) to detect generalization gaps.
"""))

cells.append(md("### 7.1 Learning Curves Computation & Plotting"))
cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lc_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.08, n_estimators=100, random_state=42, n_jobs=-1)
}

train_sizes_rel = np.linspace(0.1, 1.0, 5)

for idx, (name, model) in enumerate(lc_models.items()):
    train_sizes, train_scores, val_scores, *_ = learning_curve(
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
plt.show()
"""))

# ── Section 8: Week 5 5-Fold CV ──────────────────────────────────────────────
cells.append(md("""
## 8. 5-Fold Stratified Cross-Validation (Week 5 SOP Milestone)
Evaluating model stability and variance across 5 distinct validation folds.
"""))

cells.append(md("### 8.1 Execute 5-Fold Stratified Cross-Validation"))
cells.append(code("""
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
"""))

cells.append(md("### 8.2 5-Fold Cross-Validation Stability Table"))
cells.append(code("""
cv_df = pd.DataFrame(cv_results_list).sort_values(by='mean_f1', ascending=False).drop(columns=['mean_f1'])
display(cv_df)
"""))

# ── Section 9: Hyperparameter Tuning ─────────────────────────────────────────
cells.append(md("""
## 9. Hyperparameter Optimization with GridSearchCV (Week 5 Milestone)
Fine-tuning top-performing models (**XGBoost** and **Random Forest**) to maximize ROC-AUC.
"""))

cells.append(md("### 9.1 XGBoost Hyperparameter Optimization"))
cells.append(code("""
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

print(f"Optimal XGBoost Hyperparameters: {xgb_grid.best_params_}")
print(f"Best 3-Fold ROC-AUC: {xgb_grid.best_score_:.4f}")
"""))

cells.append(md("### 9.2 Random Forest Hyperparameter Optimization"))
cells.append(code("""
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

print(f"Optimal Random Forest Hyperparameters: {rf_grid.best_params_}")
print(f"Best 3-Fold ROC-AUC: {rf_grid.best_score_:.4f}")
"""))

# ── Section 10: Ensembles & Advanced Evaluation ──────────────────────────────
cells.append(md("""
## 10. Advanced Ensemble Modeling (Week 5 Milestone)
Constructing **Soft Voting** and **Stacking** Meta-Models to combine linear and tree learners.
"""))

cells.append(md("### 10.1 Soft Voting & Stacking Ensemble Fitting"))
cells.append(code("""
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('lr', log_reg)],
    voting='soft', n_jobs=-1
).fit(X_train_scaled, y_train)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('lr', log_reg)],
    final_estimator=LogisticRegression(), cv=3, n_jobs=-1
).fit(X_train_scaled, y_train)

print("✅ Ensembles successfully trained!")
"""))

cells.append(md("### 10.2 Final Advanced Model Benchmark Table (Hold-Out Test Set)"))
cells.append(code("""
advanced_models = {
    'Stacking Classifier': stacking_clf,
    'Soft Voting Ensemble': voting_clf,
    'Tuned XGBoost': best_xgb,
    'Gradient Boosting (GBDT)': GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42).fit(X_train_scaled, y_train),
    'Tuned Random Forest': best_rf,
    'Extra Trees': ExtraTreesClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
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
        'ROC-AUC': round(roc_auc_score(y_test, probs), 4),
        'Brier Score': round(brier_score_loss(y_test, probs), 4)
    })

adv_df = pd.DataFrame(adv_results).sort_values(by='ROC-AUC', ascending=False)
display(adv_df)
"""))

cells.append(md("### 10.3 Advanced Ensemble ROC Curves"))
cells.append(code("""
plt.figure(figsize=(10, 6))
palette_adv = sns.color_palette("deep", len(advanced_models))

for idx, (name, model) in enumerate(advanced_models.items()):
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=palette_adv[idx], lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.title('Advanced Ensembles: ROC Curve Comparison', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
"""))

# ── Section 11: Feature Importance & Threshold Optimization ──────────────────
cells.append(md("""
## 11. Feature Importance & Clinical Decision Optimization (Week 5 Milestone)
Quantifying clinical feature contributions and selecting the optimal probability threshold.
"""))

cells.append(md("### 11.1 Feature Importance: XGBoost (Gain) vs Random Forest (Gini)"))
cells.append(code("""
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
feature_names = continuous_features + categorical_features
pd.Series(best_xgb.feature_importances_, index=feature_names).sort_values().plot(kind='barh', color='#f43f5e')
plt.title('Tuned XGBoost Feature Importance (Gain)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance')

plt.subplot(1, 2, 2)
pd.Series(best_rf.feature_importances_, index=feature_names).sort_values().plot(kind='barh', color='#3b82f6')
plt.title('Tuned Random Forest Feature Importance (Gini)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance')

plt.tight_layout()
plt.show()
"""))

cells.append(md("### 11.2 Clinical Decision Threshold Sweep ($0.10$ to $0.90$)"))
cells.append(code("""
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

print(f"Optimal Threshold for F1-Score: {opt_t:.2f}")
print(f"Metrics at Threshold {opt_t:.2f}:")
print(f"  F1-Score:    {t_df.loc[best_f1_idx, 'f1']:.4f}")
print(f"  Recall/Sens: {t_df.loc[best_f1_idx, 'recall']:.4f}")
print(f"  Precision:   {t_df.loc[best_f1_idx, 'precision']:.4f}")
"""))

cells.append(md("### 11.3 Plot Decision Threshold Optimization Curve"))
cells.append(code("""
plt.figure(figsize=(10, 5))
plt.plot(t_df['threshold'], t_df['precision'], label='Precision (PPV)', color='#f59e0b', lw=2)
plt.plot(t_df['threshold'], t_df['recall'], label='Recall (Sensitivity)', color='#10b981', lw=2)
plt.plot(t_df['threshold'], t_df['f1'], label='F1-Score', color='#6366f1', lw=2.5)
plt.axvline(opt_t, color='#e11d48', linestyle=':', label=f"Optimal F1 Threshold ({opt_t:.2f})")
plt.axvline(0.50, color='#64748b', linestyle='--', label="Default (0.50)")

plt.xlabel('Classification Probability Threshold')
plt.ylabel('Score')
plt.title('Clinical Operating Threshold vs Performance Metrics', fontsize=13, fontweight='bold')
plt.legend(loc='lower center', ncol=3)
plt.grid(True, alpha=0.3)
plt.show()
"""))

# ── Section 12: Model Persistence ───────────────────────────────────────────
cells.append(md("""
## 12. Model Persistence & Deployment Artifacts
Exporting model weights, scaler coefficients, and pickled pipelines for production and frontend integration.
"""))

cells.append(md("### 12.1 Export Trained Pipeline with Joblib"))
cells.append(code("""
checkpoint_dir = "/Users/cryptorth/Documents/ml/projects"
os.makedirs(checkpoint_dir, exist_ok=True)

joblib.dump(best_xgb, os.path.join(checkpoint_dir, "best_xgboost_model.pkl"))
joblib.dump(preprocessor, os.path.join(checkpoint_dir, "preprocessor_scaler.pkl"))
print("✅ Saved best_xgboost_model.pkl and preprocessor_scaler.pkl successfully!")
"""))

# Write and execute the modular notebook
nb = nbformat.v4.new_notebook(cells=cells)
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Total Modular Cells created: {len(cells)}")
print("Now executing all cells sequentially via NotebookClient...")

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    resources={'metadata': {'path': '/Users/cryptorth/Documents/ml/projects'}}
)
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("SUCCESS: Modular notebook executed and saved with ZERO errors!")
