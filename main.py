# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# NEW: Imports for Unsupervised Learning (Distinction Requirement)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("cleaned_data.csv")

print("DATA LOADED")
print("Columns:", df.columns.tolist())

# =========================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =========================
print("GENERATING EDA PLOTS...")

# Treatment Distribution
plt.figure(figsize=(8, 5))
df["treatment"].value_counts().plot(kind="bar")
plt.title("Treatment Distribution")
plt.show()

# Gender Vs Treatment
pd.crosstab(df['Gender'], df["treatment"]).plot(kind="bar", stacked=True)
plt.title("Gender vs Treatment")
plt.tight_layout()
plt.show()

# Growing_Stress Vs Treatment
pd.crosstab(df['Growing_Stress'], df["treatment"]).plot(kind="bar", stacked=True)
plt.title("Growing_Stress vs Treatment")
plt.tight_layout()
plt.show()

# Mood_Swings Vs Treatment
pd.crosstab(df['Mood_Swings'], df["treatment"]).plot(kind="bar", stacked=True)
plt.title("Mood_Swings vs Treatment")
plt.tight_layout()
plt.show()

# Work Interest Vs Treatment
pd.crosstab(df['Work_Interest'], df["treatment"]).plot(kind="bar", stacked=True)
plt.title("Work_Interest vs Treatment")
plt.tight_layout()
plt.show()

# Correlation Heatmap
df_encoded_eda = df.copy()
for col in df_encoded_eda.columns:
    df_encoded_eda[col] = df_encoded_eda[col].astype("category").cat.codes

plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded_eda.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================
# NEW: 2.1 UNSUPERVISED LEARNING BLOCK (Distinction Requirement)
# ============================================
# This section demonstrates "Knowledge Discovery" beyond simple prediction.
print("🔍 DISCOVERING HIDDEN PATTERNS VIA CLUSTERING...")

# Using K-Means to find natural groupings in the data
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_encoded_eda)

# PCA for dimensionality reduction to visualize the clusters in 2D
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_encoded_eda)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Unsupervised Learning: Respondent Clusters (PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster ID')
plt.show()

# =========================
# 3. DEFINE TARGET (FIXED)
# =========================
target_column = "treatment"

if target_column not in df.columns:
    raise ValueError("❌ 'treatment' column NOT found. Check dataset.")

# =========================
# 4. ENCODE TARGET AND FEATURES
# =========================
# Encode target
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column].astype(str))

# Create X and y
X = df.drop(columns=[target_column])
y = df[target_column]

# One-hot encoding for categorical features (Better for Neural Nets)
X = pd.get_dummies(X, drop_first=True).astype(int)

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. MODELS
# =========================
dt = DecisionTreeClassifier(random_state=42)

xgb = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False
)

# =========================
# 7. HYPERPARAMETER TUNING
# =========================
param_dt = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

param_xgb = {
    'n_estimators': [100],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

# NEW: Added n_jobs=-1 to implement Parallel Processing (HPC Technique)
grid_dt = GridSearchCV(dt, param_dt, cv=3, scoring='f1', n_jobs=-1)
grid_xgb = GridSearchCV(xgb, param_xgb, cv=3, scoring='f1', n_jobs=-1)

grid_dt.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
best_xgb = grid_xgb.best_estimator_

print("Best DT:", grid_dt.best_params_)
print("Best XGB:", grid_xgb.best_params_)

# =========================
# 8. EVALUATION FUNCTION
# =========================
results = []

def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_prob)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc_val)

    print(classification_report(y_test, y_pred))

    results.append([name, acc, prec, rec, f1, auc_val])

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    plt.show()

# =========================
# 9. EVALUATE MODELS
# =========================
evaluate(best_dt, "Decision Tree")
evaluate(best_xgb, "XGBoost")

# =========================
# 10. FEATURE IMPORTANCE
# =========================
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[-15:] 

plt.figure(figsize=(10, 6))
plt.barh(X.columns[indices], importances[indices])
plt.title("XGBoost Feature Importance (Top 15)")
plt.tight_layout()
plt.show()

# =========================
# 11. PERMUTATION IMPORTANCE
# =========================
perm = permutation_importance(best_xgb, X_test, y_test, n_repeats=5)

idx = perm.importances_mean.argsort()[-15:] 

plt.figure(figsize=(10, 6))
plt.barh(X.columns[idx], perm.importances_mean[idx])
plt.title("Permutation Importance (Top 15)")
plt.tight_layout()
plt.show()

# =========================
# 12. DEEP LEARNING MODEL
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# NOTE: Maintained your original architecture but ensured professional execution
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 13. DL EVALUATION
# =========================
y_prob_dl = model.predict(X_test_scaled).ravel()
y_pred_dl = (y_prob_dl > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_dl)
prec = precision_score(y_test, y_pred_dl)
rec = recall_score(y_test, y_pred_dl)
f1 = f1_score(y_test, y_pred_dl)
auc_dl = roc_auc_score(y_test, y_prob_dl)

print("\nNeural Network")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("AUC:", auc_dl)

results.append(["Neural Net", acc, prec, rec, f1, auc_dl])

# =========================
# 14. FINAL RESULTS
# =========================
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"
])

print("\nFINAL RESULTS")
print(results_df)

# ============================================
# NEW: 15. SCIENTIFIC COMPARISON (Distinction Requirement)
# ============================================
# Visualizing the comparison of models using ROC Curves
print("📊 GENERATING FINAL MODEL COMPARISON...")
plt.figure(figsize=(10, 8))

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, best_dt.predict_proba(X_test)[:, 1])
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc(fpr_dt, tpr_dt):.3f})')

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, best_xgb.predict_proba(X_test)[:, 1])
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc(fpr_xgb, tpr_xgb):.3f})')

# Neural Network ROC
fpr_dl, tpr_dl, _ = roc_curve(y_test, y_prob_dl)
plt.plot(fpr_dl, tpr_dl, label=f'Neural Network (AUC = {auc(fpr_dl, tpr_dl):.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Scientific Comparison: Model ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()