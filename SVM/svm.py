import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pickle
import os

# --- Prompt for mode ---
mode = input("Select mode: (q) Quick or (f) Full GridSearch? ").strip().lower()

# --- Load data ---
train = pd.read_csv("output/train.csv")
test = pd.read_csv("output/test.csv")

# Separate features & labels
X_train = train.drop(columns=['label'])
y_train = train['label']
X_test = test.drop(columns=['label'])
y_test = test['label']

# --- Handle NaN values ---
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# --- Model Selection ---
if mode == 'f':
    print("[INFO] Running Full GridSearch (slow)...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    clf = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("SVM - Best params:", clf.best_params_)
else:
    print("[INFO] Running Quick Mode (fast)...")
    clf = SVC(C=1.0, gamma=0.01, kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)

# --- Save Model ---
os.makedirs("models", exist_ok=True)
with open("models/svm_model.sav", "wb") as f:
    pickle.dump(clf, f)

# --- Predictions ---
y_score = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

# --- ROC & EER ---
fpr, tpr, thresholds = roc_curve(y_test, y_score)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
roc_auc = auc(fpr, tpr)

print(f"SVM - Threshold for EER = {thresh}")
print(f"SVM - EER = {eer:.4f}")
print(f"SVM - AUC = {roc_auc:.4f}")

# --- Plot ROC ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('ROC Curve for SVM')
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- Save results ---
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame({'score': y_score, 'label': y_test})
results_df.to_csv("results/test_result_svm.csv", index=False)

print("Test results for SVM saved to results/test_result_svm.csv")
print("\n--- SVM Process Completed ---")
