import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def train_svm(features, labels):
    C_range = [1.0, 5.0, 10.0, 20.0, 40.0, 50]
    gamma_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['linear', 'rbf', 'poly'])
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print(f"SVM - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid

def train_knn(features, labels):
    k_range = list(range(1, 21))
    param_grid = dict(n_neighbors=k_range, weights=['uniform', 'distance'])
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print(f"KNN - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid

def train_rf(features, labels):
    tree_range = list(range(50, 200, 10))
    param_grid = dict(n_estimators=tree_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print(f"RF - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid

def evaluate(model, features, labels, model_name):
    y_prob = model.predict_proba(features)
    y_predict = model.predict(features)
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])

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
            print(f"{model_name} - EER = {eer:.4f}")
            print(f"{model_name} - AUC = {auc_result:.4f}")
            plot_roc(fpr, tpr, eer, auc_result, model_name)
            return y_prob[:, 1], y_predict
        except ValueError as e:
            print(f"Error calculating EER/AUC for {model_name}: {e}")
            print("This can happen if the ROC curve is too degenerate (e.g., all 0s or all 1s in predictions).")
            return np.array([]), np.array([])
    else:
        print(f"Warning: Could not generate ROC curve for {model_name} due to insufficient data or degenerate predictions.")
        return np.array([]), np.array([])

def normalize_train_test(train_features, test_features, save_scaler_to):
    os.makedirs(os.path.dirname(save_scaler_to), exist_ok=True)
    all_features = np.append(train_features, test_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    with open(save_scaler_to, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler mean: ", scaler.mean_)

def normalize_features(features, load_scaler_from):
    with open(load_scaler_from, 'rb') as f:
        scaler = pickle.load(f)
    return scaler.transform(features)

def train(save_res_to, features, labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method_func):
    os.makedirs(os.path.dirname(save_model_to), exist_ok=True)
    features = normalize_features(features, load_scaler_from)
    if leave_one_subject_out:
        pass
    elif cross_validation:
        pass
    else:
        model = method_func(features=features, labels=np.ravel(labels, order='C'))
        with open(save_model_to, 'wb') as f:
            pickle.dump(model, f)

def plot_roc(fpr, tpr, eer, auc_result, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_result:.2f})')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curve for {model_name}', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)

    ax.plot(eer, eer, 'o', markersize=8, color='red', label=f'EER = {eer:.2f}', zorder=5)
    ax.legend(loc="lower right", fontsize=10)

    axins = zoomed_inset_axes(ax, zoom=3, loc='upper left', bbox_to_anchor=(0.05, 0.95, 0.3, 0.3), bbox_transform=ax.transAxes)
    axins.plot(fpr, tpr, color='blue', lw=2)

    # Define the desired FPR range for the zoom box
    # You can adjust these values based on the region of the curve you want to highlight.
    # For example, to focus on the lower left corner (high specificity, low FPR):
    zoom_range_fpr_start = 0.0
    zoom_range_fpr_end = 0.1 # This will zoom from FPR 0 to 0.1

    # Find the corresponding TPR values within this FPR range
    # Interpolate TPR for the chosen FPR range
    interp_tpr = interp1d(fpr, tpr)
    zoom_tpr_at_start = interp_tpr(zoom_range_fpr_start)
    zoom_tpr_at_end = interp_tpr(zoom_range_fpr_end)

    # Set the limits for the inset axes based on the desired FPR range
    axins.set_xlim(zoom_range_fpr_start, zoom_range_fpr_end)

    # Determine the y-limits for the inset to tightly fit the curve within the zoomed x-range
    # Find the min and max TPR values within the specified FPR range
    # This assumes fpr is sorted.
    tpr_in_zoom_range = tpr[(fpr >= zoom_range_fpr_start) & (fpr <= zoom_range_fpr_end)]
    if len(tpr_in_zoom_range) > 0:
        zoom_y_min = np.min(tpr_in_zoom_range) - 0.01 # Add a small buffer
        zoom_y_max = np.max(tpr_in_zoom_range) + 0.01 # Add a small buffer
    else:
        # Fallback if no points are exactly in range (shouldn't happen with interpolation)
        zoom_y_min = 0.0
        zoom_y_max = 1.0

    axins.set_ylim(zoom_y_min, zoom_y_max)

    axins.grid(True, linestyle=':', alpha=0.7)
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.lower()}.png")
    plt.show()

def test(features, labels, load_scaler_from, load_model_from, save_res_to, model_name):
    os.makedirs(os.path.dirname(save_res_to), exist_ok=True)
    features = normalize_features(features, load_scaler_from)
    with open(load_model_from, 'rb') as f:
        model = pickle.load(f)
    y_score, y_predict = evaluate(model, features, labels, model_name)

    if y_score.size > 0:
        y_score_df = pd.DataFrame(y_score, columns=["score"])
        labels_df = pd.DataFrame(labels, columns=["label"])
        res = pd.concat([y_score_df, labels_df], axis=1)
        res.to_csv(save_res_to, index=False, header=False)
        return res
    else:
        print(f"Skipping result saving for {model_name} due to empty scores.")
        return pd.DataFrame()

if __name__ == '__main__':
    is_normalized = False
    train_model = True
    cross_validation = False
    leave_one_subject_out = False

    output_base_dir = './res'
    models_dir = os.path.join(output_base_dir, 'models')
    training_events_dir = os.path.join(output_base_dir, 'training_events')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(training_events_dir, exist_ok=True)

    save_scaler_to = os.path.join(models_dir, '_scaler.sav')

    model_filenames = {
        'svm': os.path.join(training_events_dir, 'svm_model.sav'),
        'knn': os.path.join(training_events_dir, 'knn_model.sav'),
        'rf':  os.path.join(training_events_dir, 'rf_model.sav')
    }

    save_result_files = {
        'svm': './test_result_svm.csv',
        'knn': './test_result_knn.csv',
        'rf':  './test_result_rf.csv'
    }

    input_train_data = pd.read_csv('./output/train.csv')
    input_train_data = input_train_data.fillna(0)
    train_features = np.array(input_train_data.iloc[:, :-1])
    train_labels = np.array(input_train_data.iloc[:, -1]).astype(int).reshape((-1, 1))

    input_test_data = pd.read_csv('./output/test.csv')
    input_test_data = input_test_data.fillna(0)
    test_features = np.array(input_test_data.iloc[:, :-1])
    test_labels = np.array(input_test_data.iloc[:, -1]).astype(int).reshape((-1, 1))

    if not is_normalized:
        print("Performing normalization and saving scaler...")
        normalize_train_test(train_features, test_features, save_scaler_to)
        is_normalized = True

    if train_model:
        models_to_train = [
            (train_svm, 'svm'),
            (train_knn, 'knn'),
            (train_rf, 'rf')
        ]
        for train_func, model_key in models_to_train:
            print(f"\n--- Training {model_key.upper()} ---")
            train('', train_features, train_labels, save_scaler_to, model_filenames[model_key],
                  leave_one_subject_out, cross_validation, train_func)

    print("\n=== Testing all models ===")
    final_results = {}
    for method_name in ['svm', 'knn', 'rf']:
        print(f"\n--- Testing {method_name.upper()} ---")
        res = test(test_features, test_labels, save_scaler_to, model_filenames[method_name],
                   save_result_files[method_name], method_name)
        final_results[method_name] = res

    for name, df in final_results.items():
        if not df.empty:
            print("*******************************************")
            print(f"Results for: {name.upper()}")
            print(df.head())
        else:
            print(f"*******************************************")
            print(f"No results to display for: {name.upper()}")