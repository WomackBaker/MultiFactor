# Import necessary libraries for machine learning, data processing, and visualization
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

# Configure matplotlib font sizes for better readability
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# Define file paths and directories for data and model storage
TRAIN_CSV_PATH = "output/train.csv"    # Path to training dataset
TEST_CSV_PATH = "output/test.csv"      # Path to testing dataset
MODEL_OUTPUT_DIR = "models"            # Directory to save trained models
RESULTS_OUTPUT_DIR = "results"         # Directory to save results

def train_svm(features, labels):
    """
    Train an SVM model using GridSearchCV for hyperparameter optimization.
    
    Args:
        features (array): Training feature matrix
        labels (array): Training labels
    
    Returns:
        GridSearchCV: Trained SVM model with best hyperparameters
    """
    # Define hyperparameter search space
    C_range = [1.0, 10.0]              # Regularization parameter values
    gamma_range = [0.01, 0.1]          # Kernel coefficient values
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'])

    # Set up stratified cross-validation to ensure balanced class distribution
    # Set up stratified cross-validation to ensure balanced class distribution
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.20, random_state=0)
    
    # Perform grid search with cross-validation to find optimal hyperparameters
    print("Starting GridSearchCV for SVM...")
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1, verbose=1)
    grid.fit(features, labels)
    
    # Display the best parameters and score found
    print(f"SVM - Best params: {grid.best_params_}, CV score={grid.best_score_:.2f}")
    return grid

def evaluate(model, features, labels, model_name):
    """
    Evaluate the trained model and calculate performance metrics including EER and AUC.
    
    Args:
        model: Trained machine learning model
        features (array): Test feature matrix
        labels (array): True test labels
        model_name (str): Name of the model for logging purposes
    
    Returns:
        tuple: (predicted_probabilities, predicted_labels)
    """
    # Get prediction probabilities and class predictions
    y_prob = model.predict_proba(features)
    y_predict = model.predict(features)
    
    # Calculate ROC curve values (False Positive Rate, True Positive Rate, thresholds)
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])

    # Calculate Equal Error Rate (EER) - point where FPR = FNR
    err = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(err)
    
    # Save threshold to results file
    # Save threshold to results file
    with open('results.txt', 'a') as f:
        f.write(f"{thresh}\n")
    
    # Validate and process ROC curve data
    if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0:
        # Ensure ROC curve starts at (0,0)
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
        # Ensure ROC curve ends at (1,1)
        if fpr[-1] != 1 or tpr[-1] != 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        # Remove duplicate FPR values to avoid interpolation issues
        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        fpr = unique_fpr
        tpr = tpr[unique_indices]
        
        try:
            # Calculate Equal Error Rate and Area Under Curve
            # Calculate Equal Error Rate and Area Under Curve
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            auc_result = auc(fpr, tpr)
            
            # Print and save performance metrics
            print(f"{model_name} - Threshold for EER = {thresh}")
            print(f"{model_name} - EER = {eer:.4f}")
            print(f"{model_name} - AUC = {auc_result:.4f}")
            with open('results.txt', 'a') as f:
                f.write(f"{eer:.4f}\n")
                f.write(f"{auc_result:.4f}\n")

            # Generate and save ROC curve plot
            plot_roc(fpr, tpr, eer, auc_result, model_name)
            return y_prob[:, 1], y_predict
        except ValueError as e:
            print(f"Error calculating EER/AUC for {model_name}: {e}")
            return np.array([]), np.array([])
    else:
        print(f"Warning: Could not generate ROC curve for {model_name} due to insufficient data or degenerate predictions.")
        return np.array([]), np.array([])

def plot_roc(fpr, tpr, eer, auc_result, model_name):
    """
    Generate and save a ROC curve plot with zoomed inset for detailed view.
    
    Args:
        fpr (array): False Positive Rate values
        tpr (array): True Positive Rate values
        eer (float): Equal Error Rate value
        auc_result (float): Area Under Curve value
        model_name (str): Name of the model for plot title and filename
    """
    # Create main ROC curve plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_result:.2f})')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Diagonal reference line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curve for {model_name}', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)

    # Add EER point to the plot
    ax.plot(eer, eer, label=f'EER = {eer:.2f}')
    ax.legend(loc="lower right", fontsize=10)

    # Create zoomed inset for detailed view of the upper-left region
    axins = zoomed_inset_axes(ax, zoom=2, loc='center', bbox_to_anchor=(0.5, 0.5, 0.2, 0.2), bbox_transform=ax.transAxes)
    axins.plot(fpr, tpr, color='blue', lw=2)

    # Set zoom region to focus on high-performance area
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.8, 1.0)

    # Style the inset plot
    axins.grid(True, linestyle=':', alpha=0.7)
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=8)

    # Add lines connecting the inset to the main plot
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", lw=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name.lower()}.png")
    plt.show()

def fit_and_save_scaler(train_features, save_scaler_to):
    """
    Fit a StandardScaler on training features and save it for later use.
    StandardScaler normalizes features to have mean=0 and std=1.
    
    Args:
        train_features (array): Training feature matrix
        save_scaler_to (str): Path to save the fitted scaler
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_scaler_to), exist_ok=True)
    
    # Fit the scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Save the fitted scaler for consistent transformation of test data
    with open(save_scaler_to, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler fitted and saved.")
    print("Scaler mean: ", scaler.mean_)
    print("Scaler scale (std dev): ", scaler.scale_)

def load_and_transform_features(features, load_scaler_from):
    """
    Load a previously fitted scaler and transform features using it.
    This ensures consistent normalization between training and test data.
    
    Args:
        features (array): Feature matrix to transform
        load_scaler_from (str): Path to the saved scaler file
    
    Returns:
        array: Transformed (normalized) features
    """
    # Load the previously fitted scaler
    with open(load_scaler_from, 'rb') as f:
        scaler = pickle.load(f)
    
    # Transform features using the loaded scaler
    return scaler.transform(features)

def run_training_and_testing(train_features, train_labels, test_features, test_labels,
                             scaler_path, model_path, result_path, model_name):
    """
    Complete pipeline for training and testing an SVM model.
    
    Args:
        train_features (array): Training feature matrix
        train_labels (array): Training labels
        test_features (array): Test feature matrix
        test_labels (array): Test labels
        scaler_path (str): Path to save/load the feature scaler
        model_path (str): Path to save the trained model
        result_path (str): Path to save test results
        model_name (str): Name of the model for logging
    """
    # Create output directories if they don't exist
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Step 1: Normalize the data using StandardScaler
    print("\n--- Normalizing Data ---")
    fit_and_save_scaler(train_features, scaler_path)
    
    # Transform both training and test features using the same scaler
    print("Transforming training features...")
    scaled_train_features = load_and_transform_features(train_features, scaler_path)
    print("Transforming testing features...")
    scaled_test_features = load_and_transform_features(test_features, scaler_path)

    # Step 2: Train the SVM model with hyperparameter optimization
    print(f"\n--- Training {model_name} Model ---")
    trained_model = train_svm(features=scaled_train_features, labels=np.ravel(train_labels))
    
    # Save the trained model for future use
    with open(model_path, 'wb') as f:
        pickle.dump(trained_model, f)
    print(f"{model_name} model saved to {model_path}")

    # Step 3: Evaluate the model on test data
    print(f"\n--- Testing {model_name} Model ---")
    y_score, y_predict = evaluate(trained_model, scaled_test_features, test_labels, model_name)

    # Step 4: Save test results if evaluation was successful
    if y_score.size > 0:
        y_score_df = pd.DataFrame(y_score, columns=["score"])
        labels_df = pd.DataFrame(test_labels, columns=["label"])
        res_df = pd.concat([y_score_df, labels_df], axis=1)
        res_df.to_csv(result_path, index=False, header=False)
        print(f"Test results for {model_name} saved to {result_path}")
    else:
        print(f"Skipping result saving for {model_name} due to empty scores.")

if __name__ == '__main__':
    """
    Main execution block - loads data, trains SVM model, and evaluates performance.
    This script expects pre-split training and testing CSV files from split.py.
    """
    # Define file paths for model and results storage
    scaler_save_path = os.path.join(MODEL_OUTPUT_DIR, '_scaler.sav')
    svm_model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'svm_model.sav')
    svm_results_save_path = os.path.join(RESULTS_OUTPUT_DIR, 'test_result_svm.csv')

    # Ensure output directories exist
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # Load and prepare training data
    print(f"Loading training data from {TRAIN_CSV_PATH}...")
    try:
        input_train_data = pd.read_csv(TRAIN_CSV_PATH)
        input_train_data = input_train_data.fillna(0)  # Fill missing values with 0
        
        # Separate features (all columns except last) and labels (last column)
        train_features = input_train_data.iloc[:, :-1].values
        train_labels = input_train_data.iloc[:, -1].values.astype(int).reshape((-1, 1))
        print(f"Training data loaded. Features shape: {train_features.shape}, Labels shape: {train_labels.shape}")
    except FileNotFoundError:
        print(f"Error: Training file not found at '{TRAIN_CSV_PATH}'. Please ensure split.py has run.")
        exit()

    # Load and prepare testing data
    print(f"Loading testing data from {TEST_CSV_PATH}...")
    try:
        input_test_data = pd.read_csv(TEST_CSV_PATH)
        input_test_data = input_test_data.fillna(0)  # Fill missing values with 0
        
        # Separate features (all columns except last) and labels (last column)
        test_features = input_test_data.iloc[:, :-1].values
        test_labels = input_test_data.iloc[:, -1].values.astype(int).reshape((-1, 1))
        print(f"Testing data loaded. Features shape: {test_features.shape}, Labels shape: {test_labels.shape}")
    except FileNotFoundError:
        print(f"Error: Testing file not found at '{TEST_CSV_PATH}'. Please ensure split.py has run.")
        exit()

    # Execute the complete training and testing pipeline
    run_training_and_testing(train_features, train_labels, test_features, test_labels,
                             scaler_save_path, svm_model_save_path, svm_results_save_path, 'SVM')

    print("\n--- SVM Process Completed ---")