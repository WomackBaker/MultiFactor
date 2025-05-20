import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV # Keep GridSearchCV for SVM
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os

# Set global matplotlib font sizes for better readability
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def train_svm(features, labels):
    """
    Trains an SVM model using GridSearchCV for hyperparameter tuning.
    The parameter grid has been reduced for faster execution.
    """
    # Reduced parameter ranges for faster training as discussed previously
    C_range = [1.0, 10.0] # Example: narrowed down C values
    gamma_range = [0.01, 0.1] # Example: narrowed down gamma values
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf']) # Example: focusing on 'rbf' kernel

    # Reduced number of cross-validation splits for faster tuning
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.20, random_state=0)
    
    print("Starting GridSearchCV for SVM...")
    # Initialize GridSearchCV with SVC and the defined parameter grid and cross-validation strategy
    # n_jobs=-1 uses all available CPU cores for parallel processing
    grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels) # Fit the GridSearchCV to find the best parameters
    
    # Print the best parameters found and the corresponding cross-validation score
    print(f"SVM - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid # Return the fitted GridSearchCV object (which contains the best estimator)

# Removed train_knn and train_rf functions as they are no longer needed.

def evaluate(model, features, labels, model_name):
    """
    Evaluates the trained model, calculates ROC curve, AUC, and EER,
    and plots the ROC curve.
    """
    # Predict probabilities for the positive class (class 1)
    y_prob = model.predict_proba(features)
    # Predict class labels
    y_predict = model.predict(features)
    
    # Calculate False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])

    # Ensure ROC curve starts at (0,0) and ends at (1,1) for proper plotting and AUC calculation
    if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0:
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
        if fpr[-1] != 1 or tpr[-1] != 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        # Remove duplicate FPR values to ensure interpolation works correctly
        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        fpr = unique_fpr
        tpr = tpr[unique_indices]
        
        try:
            # Calculate Equal Error Rate (EER) where FPR = 1 - TPR
            # brentq finds the root of a function within an interval
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            # Calculate Area Under the Curve (AUC)
            auc_result = auc(fpr, tpr)
            
            print(f"{model_name} - EER = {eer:.4f}")
            print(f"{model_name} - AUC = {auc_result:.4f}")
            
            # Plot the ROC curve
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
    """
    Fits a StandardScaler on combined training and testing features and saves it.
    """
    os.makedirs(os.path.dirname(save_scaler_to), exist_ok=True)
    all_features = np.append(train_features, test_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features) # Fit the scaler on all data to prevent data leakage from test set
    with open(save_scaler_to, 'wb') as f:
        pickle.dump(scaler, f) # Save the fitted scaler
    print("Scaler mean: ", scaler.mean_)

def normalize_features(features, load_scaler_from):
    """
    Loads a saved StandardScaler and transforms the given features.
    """
    with open(load_scaler_from, 'rb') as f:
        scaler = pickle.load(f) # Load the previously saved scaler
    return scaler.transform(features) # Transform features using the loaded scaler

def train(save_res_to, features, labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method_func):
    """
    Orchestrates the training process for a given model.
    """
    os.makedirs(os.path.dirname(save_model_to), exist_ok=True)
    features = normalize_features(features, load_scaler_from) # Normalize features before training
    
    # Placeholder for more complex training scenarios (not implemented in this simplified version)
    if leave_one_subject_out:
        print("Leave-one-subject-out cross-validation is not implemented in this script.")
        pass
    elif cross_validation:
        print("General cross-validation is handled by GridSearchCV within train_svm.")
        pass
    else:
        # Call the specific training function (e.g., train_svm)
        model = method_func(features=features, labels=np.ravel(labels, order='C'))
        with open(save_model_to, 'wb') as f:
            pickle.dump(model, f) # Save the trained model

def plot_roc(fpr, tpr, eer, auc_result, model_name):
    """
    Plots the ROC curve with AUC and EER marked, including a zoomed inset
    focusing on the top-left (low FPR, high TPR) region.
    """
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

    # Plot EER point
    ax.plot(eer, eer, 'o', markersize=8, color='red', label=f'EER = {eer:.2f}', zorder=5)
    ax.legend(loc="lower right", fontsize=10)

    # Create a zoomed inset for the top-left region of the ROC curve
    # Adjusted 'zoom' factor and 'bbox_to_anchor' for smaller size and middle placement
    axins = zoomed_inset_axes(ax, zoom=2, loc='center', bbox_to_anchor=(0.5, 0.5, 0.2, 0.2), bbox_transform=ax.transAxes)
    axins.plot(fpr, tpr, color='blue', lw=2)
    # Optionally, you can still plot the EER point in the inset if desired, but the focus is the curve
    # axins.plot(eer, eer, 'o', markersize=8, color='red', zorder=5)

    # Set x and y limits to zoom into the top-left corner (low FPR, high TPR)
    axins.set_xlim(0, 0.2)   # FPR from 0 to 0.2
    axins.set_ylim(0.8, 1.0) # TPR from 0.8 to 1.0

    axins.grid(True, linestyle=':', alpha=0.7)
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)

    # Adjusted 'loc1' and 'loc2' for mark_inset to connect to the new inset position
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.lower()}.png") # Save the plot
    plt.show() # Display the plot

def test(features, labels, load_scaler_from, load_model_from, save_res_to, model_name):
    """
    Tests a trained model, evaluates its performance, and saves the results.
    """
    os.makedirs(os.path.dirname(save_res_to), exist_ok=True)
    features = normalize_features(features, load_scaler_from) # Normalize test features
    with open(load_model_from, 'rb') as f:
        model = pickle.load(f) # Load the trained model
    
    # Evaluate the model and get scores/predictions
    y_score, y_predict = evaluate(model, features, labels, model_name)

    if y_score.size > 0:
        # Save scores and labels to a CSV file
        y_score_df = pd.DataFrame(y_score, columns=["score"])
        labels_df = pd.DataFrame(labels, columns=["label"])
        res = pd.concat([y_score_df, labels_df], axis=1)
        res.to_csv(save_res_to, index=False, header=False)
        return res
    else:
        print(f"Skipping result saving for {model_name} due to empty scores.")
        return pd.DataFrame()

if __name__ == '__main__':
    is_normalized = False # Flag to check if data has been normalized
    train_model = True # Flag to control model training
    cross_validation = False # Not directly used for overall script flow, GridSearchCV handles it
    leave_one_subject_out = False # Not implemented in this script

    output_base_dir = './res'
    models_dir = os.path.join(output_base_dir, 'models')
    training_events_dir = os.path.join(output_base_dir, 'training_events')

    # Create necessary output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(training_events_dir, exist_ok=True)

    save_scaler_to = os.path.join(models_dir, '_scaler.sav')

    # Define filenames for the SVM model and its results
    model_filenames = {
        'svm': os.path.join(training_events_dir, 'svm_model.sav')
    }

    save_result_files = {
        'svm': './test_result_svm.csv'
    }

    # Load training data
    print("Loading training data...")
    input_train_data = pd.read_csv('./output/train.csv')
    input_train_data = input_train_data.fillna(0) # Handle potential NaN values
    train_features = np.array(input_train_data.iloc[:, :-1]) # All columns except the last for features
    train_labels = np.array(input_train_data.iloc[:, -1]).astype(int).reshape((-1, 1)) # Last column for labels

    # Load testing data
    print("Loading testing data...")
    input_test_data = pd.read_csv('./output/test.csv')
    input_test_data = input_test_data.fillna(0) # Handle potential NaN values
    test_features = np.array(input_test_data.iloc[:, :-1]) # All columns except the last for features
    test_labels = np.array(input_test_data.iloc[:, -1]).astype(int).reshape((-1, 1)) # Last column for labels

    # Perform normalization if not already done
    if not is_normalized:
        print("Performing normalization and saving scaler...")
        normalize_train_test(train_features, test_features, save_scaler_to)
        is_normalized = True # Set flag to true after normalization

    if train_model:
        # Only train the SVM model
        print(f"\n--- Training SVM ---")
        train('', train_features, train_labels, save_scaler_to, model_filenames['svm'],
              leave_one_subject_out, cross_validation, train_svm)

    print("\n=== Testing SVM model ===")
    final_results = {}
    # Only test the SVM model
    print(f"\n--- Testing SVM ---")
    res = test(test_features, test_labels, save_scaler_to, model_filenames['svm'],
                save_result_files['svm'], 'svm')
    final_results['svm'] = res

    # Display results for SVM
    if not final_results['svm'].empty:
        print("*******************************************")
        print(f"Results for: SVM")
        print(final_results['svm'].head())
    else:
        print(f"*******************************************")
        print(f"No results to display for: SVM")