# FRUIT QUALITY PREDICTION SYSTEM - COMPLETE ML PROJECT
# Dataset: Banana Quality Dataset (8000 samples)
# Problem Type: Binary Classification (Good / Bad)

# ─────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# STEP 2: LOAD & EXPLORE THE DATASET
# =============================================================================
print("=" * 60)
print("STEP 2: LOADING & EXPLORING THE DATASET")
print("=" * 60)

import os

file_path = os.path.join("data", "banana_quality.csv")

if not os.path.exists(file_path):
    print("\n❌ Dataset not found!")
    print("👉 Download from:")
    print("https://www.kaggle.com/datasets/seryoh67/banana-quality")
    print("👉 Place it inside: data/ folder\n")
    exit()

df = pd.read_csv("data/banana_quality.csv")
# df = pd.read_csv(file_path)

print("\n📌 Shape of Dataset:", df.shape)
print("\n📌 First 5 Rows:\n", df.head())
print("\n📌 Data Types:\n", df.dtypes)
print("\n📌 Basic Statistics:\n", df.describe())
print("\n📌 Missing Values:\n", df.isnull().sum())
print("\n📌 Target Class Distribution:\n", df['Quality'].value_counts())


# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

features = ['Size', 'Weight', 'Sweetness', 'Softness',
            'HarvestTime', 'Ripeness', 'Acidity']

# --- Plot 1: Target Distribution ---
plt.figure(figsize=(6, 4))
df['Quality'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.title('Target Class Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot1_class_distribution.png', dpi=150)
plt.show()
print("✅ Plot 1 saved: class distribution")

# --- Plot 2: Feature Distributions ---
df[features].hist(bins=30, figsize=(14, 8), color='steelblue', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('plot2_feature_distributions.png', dpi=150)
plt.show()
print("✅ Plot 2 saved: feature distributions")

# --- Plot 3: Correlation Heatmap ---
plt.figure(figsize=(9, 6))
sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot3_correlation_heatmap.png', dpi=150)
plt.show()
print("✅ Plot 3 saved: correlation heatmap")

# --- Plot 4: Boxplots by Quality ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feature in enumerate(features):
    df.boxplot(column=feature, by='Quality', ax=axes[i], patch_artist=True)
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
axes[-1].set_visible(False)
plt.suptitle('Feature Distributions by Quality', fontsize=13)
plt.tight_layout()
plt.savefig('plot4_boxplots.png', dpi=150)
plt.show()
print("✅ Plot 4 saved: boxplots by quality")


# =============================================================================
# STEP 4: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

# 4a. Drop missing values (if any)
df.dropna(inplace=True)
print(f"✅ After dropping NaN rows: {df.shape}")

# 4b. Encode target variable: Good → 1, Bad → 0
le = LabelEncoder()
df['Quality_encoded'] = le.fit_transform(df['Quality'])
# df['Quality_encoded'] = df['Quality'].map({'Bad': 0, 'Good': 1})
print(df[['Quality', 'Quality_encoded']].drop_duplicates())

# 4c. Define features (X) and target (y)
X = df[features]
y = df['Quality_encoded']

# 4d. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# 4e. Feature Scaling (StandardScaler — zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✅ Feature scaling applied (StandardScaler)")


# =============================================================================
# STEP 5: MODEL TRAINING & EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: TRAINING MULTIPLE ML MODELS")
print("=" * 60)

models = {
    "Logistic Regression"     : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"           : DecisionTreeClassifier(random_state=42),
    "Random Forest"           : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"       : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM"                     : SVC(kernel='rbf', probability=True, random_state=42),
    "K-Nearest Neighbors"     : KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in models.items():
    print(f"\n▶ Training: {name}")
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'model'    : model,
        'accuracy' : acc,
        'roc_auc'  : roc_auc,
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
    }

    print(f"   Accuracy  : {acc:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =============================================================================
# STEP 6: MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model'    : list(results.keys()),
    'Accuracy' : [results[m]['accuracy'] for m in results],
    'ROC-AUC'  : [results[m]['roc_auc']  for m in results],
    'CV Mean'  : [results[m]['cv_mean']  for m in results],
    'CV Std'   : [results[m]['cv_std']   for m in results],
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n", comparison_df.to_string(index=False))

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c']

axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'],
             color=colors, edgecolor='black')
axes[0].set_title('Model Accuracy Comparison', fontsize=13)
axes[0].set_xlabel('Accuracy')
axes[0].set_xlim(0.5, 1.0)
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=9)

axes[1].barh(comparison_df['Model'], comparison_df['ROC-AUC'],
             color=colors, edgecolor='black')
axes[1].set_title('Model ROC-AUC Comparison', fontsize=13)
axes[1].set_xlabel('ROC-AUC')
axes[1].set_xlim(0.5, 1.0)
for i, v in enumerate(comparison_df['ROC-AUC']):
    axes[1].text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plot5_model_comparison.png', dpi=150)
plt.show()
print("✅ Plot 5 saved: model comparison")


# =============================================================================
# STEP 7: BEST MODEL — DETAILED EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: BEST MODEL DETAILED EVALUATION")
print("=" * 60)

best_name = comparison_df.iloc[0]['Model']
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name}")

print("\n📌 Classification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot6_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Plot 6 saved: confusion matrix")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best['y_proba'])
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {best["roc_auc"]:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve — {best_name}')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plot7_roc_curve.png', dpi=150)
plt.show()
print("✅ Plot 7 saved: ROC curve")


# =============================================================================
# STEP 8: FEATURE IMPORTANCE (Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: FEATURE IMPORTANCE")
print("=" * 60)

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot8_feature_importance.png', dpi=150)
plt.show()
print("✅ Plot 8 saved: feature importance")
print("\n📌 Feature Importances:\n", importances.sort_values(ascending=False))


# =============================================================================
# STEP 9: HYPERPARAMETER TUNING (Best Model — Random Forest example)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: HYPERPARAMETER TUNING (Random Forest)")
print("=" * 60)

param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Params  : {grid_search.best_params_}")
print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")

best_tuned = grid_search.best_estimator_
y_pred_tuned = best_tuned.predict(X_test_scaled)
print(f"✅ Tuned Test Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")


# =============================================================================
# STEP 10: SAVE MODEL & MAKE PREDICTIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: SAVING MODEL & PREDICTIONS")
print("=" * 60)

import joblib

joblib.dump(best_tuned, 'best_model.pkl')
joblib.dump(scaler,     'scaler.pkl')
joblib.dump(le,         'label_encoder.pkl')
print("✅ Model, Scaler, and LabelEncoder saved as .pkl files")

# --- Predict on new data ---
print("\n📌 Sample Prediction on New Data:")
sample = pd.DataFrame([[1.2, 0.5, 2.1, -1.0, 0.3, 1.8, 0.9]], columns=features)
sample_scaled = scaler.transform(sample)
prediction    = best_tuned.predict(sample_scaled)
probability   = best_tuned.predict_proba(sample_scaled)

label = le.inverse_transform(prediction)[0]
print(f"   Input    : {sample.values.tolist()[0]}")
print(f"   Prediction: {label}")
print(f"   Probability [Bad, Good]: {probability[0].round(4).tolist()}")

print("\n" + "=" * 60)
print("✅ PROJECT COMPLETE!")
print("=" * 60)