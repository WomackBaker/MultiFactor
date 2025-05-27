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

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

TRAIN_CSV_PATH = "output/train.csv"
TEST_CSV_PATH = "output/test.csv"
MODEL_OUTPUT_DIR = "models"
RESULTS_OUTPUT_DIR = "results"

def train_svm(features, labels):
    C_range = [1.0, 10.0]
    gamma_range = [0.01, 0.1]
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'])

    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.20, random_state=0)
    
    print("Starting GridSearchCV for SVM...")
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1, verbose=1)
    grid.fit(features, labels)
    
    print(f"SVM - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid

def evaluate(model, features, labels, model_name):
    y_prob = model.predict_proba(features)
    y_predict = model.predict(features)
    
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])

    err = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(err)
    
    with open('results.txt', 'a') as f:
        f.write(f"{thresh}\n")
    
    if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0:
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
        if fpr[-1] != 1 or tpr[-1] != 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        fpr = unique_fpr
        tpr = tpr[unique_indices]
        
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            auc_result = auc(fpr, tpr)
            print(f"{model_name} - Threshold for EER = {thresh}")
            print(f"{model_name} - EER = {eer:.4f}")
            print(f"{model_name} - AUC = {auc_result:.4f}")
            with open('results.txt', 'a') as f:
                f.write(f"{eer:.4f}\n")
                f.write(f"{auc_result:.4f}\n")

            plot_roc(fpr, tpr, eer, auc_result, model_name)
            return y_prob[:, 1], y_predict
        except ValueError as e:
            print(f"Error calculating EER/AUC for {model_name}: {e}")
            return np.array([]), np.array([])
    else:
        print(f"Warning: Could not generate ROC curve for {model_name} due to insufficient data or degenerate predictions.")
        return np.array([]), np.array([])

def plot_roc(fpr, tpr, eer, auc_result, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_result:.2f})')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curve for {model_name}', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)

    ax.plot(eer, eer, label=f'EER = {eer:.2f}')
    ax.legend(loc="lower right", fontsize=10)

    axins = zoomed_inset_axes(ax, zoom=2, loc='center', bbox_to_anchor=(0.5, 0.5, 0.2, 0.2), bbox_transform=ax.transAxes)
    axins.plot(fpr, tpr, color='blue', lw=2)

    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.8, 1.0)

    axins.grid(True, linestyle=':', alpha=0.7)
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.lower()}.png")
    plt.show()

def fit_and_save_scaler(train_features, save_scaler_to):
    os.makedirs(os.path.dirname(save_scaler_to), exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(train_features)
    with open(save_scaler_to, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler fitted and saved.")
    print("Scaler mean: ", scaler.mean_)
    print("Scaler scale (std dev): ", scaler.scale_)

def load_and_transform_features(features, load_scaler_from):
    with open(load_scaler_from, 'rb') as f:
        scaler = pickle.load(f)
    return scaler.transform(features)

def run_training_and_testing(train_features, train_labels, test_features, test_labels,
                             scaler_path, model_path, result_path, model_name):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print("\n--- Normalizing Data ---")
    fit_and_save_scaler(train_features, scaler_path)
    
    print("Transforming training features...")
    scaled_train_features = load_and_transform_features(train_features, scaler_path)
    print("Transforming testing features...")
    scaled_test_features = load_and_transform_features(test_features, scaler_path)

    print(f"\n--- Training {model_name} Model ---")
    trained_model = train_svm(features=scaled_train_features, labels=np.ravel(train_labels))
    
    with open(model_path, 'wb') as f:
        pickle.dump(trained_model, f)
    print(f"{model_name} model saved to {model_path}")

    print(f"\n--- Testing {model_name} Model ---")
    y_score, y_predict = evaluate(trained_model, scaled_test_features, test_labels, model_name)

    if y_score.size > 0:
        y_score_df = pd.DataFrame(y_score, columns=["score"])
        labels_df = pd.DataFrame(test_labels, columns=["label"])
        res_df = pd.concat([y_score_df, labels_df], axis=1)
        res_df.to_csv(result_path, index=False, header=False)
        print(f"Test results for {model_name} saved to {result_path}")
    else:
        print(f"Skipping result saving for {model_name} due to empty scores.")

if __name__ == '__main__':
    scaler_save_path = os.path.join(MODEL_OUTPUT_DIR, '_scaler.sav')
    svm_model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'svm_model.sav')
    svm_results_save_path = os.path.join(RESULTS_OUTPUT_DIR, 'test_result_svm.csv')

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    print(f"Loading training data from {TRAIN_CSV_PATH}...")
    try:
        input_train_data = pd.read_csv(TRAIN_CSV_PATH)
        input_train_data = input_train_data.fillna(0) 
        train_features = input_train_data.iloc[:, :-1].values
        train_labels = input_train_data.iloc[:, -1].values.astype(int).reshape((-1, 1))
        print(f"Training data loaded. Features shape: {train_features.shape}, Labels shape: {train_labels.shape}")
    except FileNotFoundError:
        print(f"Error: Training file not found at '{TRAIN_CSV_PATH}'. Please ensure split.py has run.")
        exit()

    print(f"Loading testing data from {TEST_CSV_PATH}...")
    try:
        input_test_data = pd.read_csv(TEST_CSV_PATH)
        input_test_data = input_test_data.fillna(0)
        test_features = input_test_data.iloc[:, :-1].values
        test_labels = input_test_data.iloc[:, -1].values.astype(int).reshape((-1, 1))
        print(f"Testing data loaded. Features shape: {test_features.shape}, Labels shape: {test_labels.shape}")
    except FileNotFoundError:
        print(f"Error: Testing file not found at '{TEST_CSV_PATH}'. Please ensure split.py has run.")
        exit()

    run_training_and_testing(train_features, train_labels, test_features, test_labels,
                             scaler_save_path, svm_model_save_path, svm_results_save_path, 'SVM')

    print("\n--- SVM Process Completed ---")