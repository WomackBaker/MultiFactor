import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os

TRAIN_CSV_PATH = "output/train.csv"
TEST_CSV_PATH = "output/test.csv"
MODEL_OUTPUT_DIR = "models"
RESULTS_OUTPUT_DIR = "results"

def plot_roc(fpr, tpr, eer, auc_result, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_result:.2f})')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax.plot(eer, eer, label=f'EER = {eer:.2f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'ROC Curve for {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True)

    axins = zoomed_inset_axes(ax, zoom=2, loc='center',
                              bbox_to_anchor=(0.5, 0.5, 0.2, 0.2),
                              bbox_transform=ax.transAxes)
    axins.plot(fpr, tpr, color='blue', lw=2)
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.8, 1.0)
    axins.grid(True, linestyle=':', alpha=0.7)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", lw=0.5)

    plt.tight_layout()
    plt.show()

def evaluate(model, features, labels, model_name):
    y_prob = model.decision_function(features)
    y_score = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else y_prob
    y_predict = model.predict(features)

    fpr, tpr, thresholds = roc_curve(labels, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    auc_result = auc(fpr, tpr)

    print(f"[DEBUG] Mean predicted probability for class 1: {np.mean(y_score)}")
    print(f"{model_name} - Threshold for EER = {thresh}")
    print(f"{model_name} - EER = {eer:.4f}")
    print(f"{model_name} - AUC = {auc_result:.4f}")

    plot_roc(fpr, tpr, eer, auc_result, model_name)
    return y_score, y_predict

if __name__ == '__main__':
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV_PATH).fillna(0)
    test_df = pd.read_csv(TEST_CSV_PATH).fillna(0)
    train_features = train_df.iloc[:, :-1].values
    train_labels = train_df.iloc[:, -1].values
    test_features = test_df.iloc[:, :-1].values
    test_labels = test_df.iloc[:, -1].values

    # Scaling
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Optional noise injection for realism
    train_features += np.random.normal(0, 0.01, train_features.shape)

    # Hyperparameter search
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, verbose=1, n_jobs=-1)
    grid.fit(train_features, train_labels)

    print(f"SVM - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    model = grid.best_estimator_

    # Save model
    pickle.dump(model, open(os.path.join(MODEL_OUTPUT_DIR, 'svm_model.sav'), 'wb'))

    # Evaluate
    y_score, y_predict = evaluate(model, test_features, test_labels, "SVM")

    # Check predicted label distribution
    print("\n[DEBUG] Predicted labels distribution:")
    print(pd.Series(y_predict).value_counts())

    # Feature importance for linear kernel
    if model.kernel == 'linear':
        importance = np.abs(model.coef_[0])
        print("\n[DEBUG] Feature importance (linear kernel):")
        for col, val in sorted(zip(train_df.columns[:-1], importance), key=lambda x: -x[1]):
            print(f"{col}: {val:.4f}")