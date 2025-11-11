"""
FPR (False Positive Rate) and FNR (False Negative Rate) Calculator

This script calculates and displays classification metrics including FPR and FNR
for a given threshold value. It loads test results from an SVM model and 
evaluates performance against true labels.

Metrics calculated:
- True Positives (TP): Correctly predicted positive cases
- True Negatives (TN): Correctly predicted negative cases  
- False Positives (FP): Incorrectly predicted positive cases
- False Negatives (FN): Incorrectly predicted negative cases
- FPR: False Positive Rate = FP / (FP + TN)
- FNR: False Negative Rate = FN / (FN + TP)
"""

import pandas as pd
import numpy as np

# Path to the CSV file containing SVM test results (scores and true labels)
INPUT_RESULTS_CSV = "results/test_result_svm.csv"

# Read the threshold value from results.txt file
# This threshold is typically determined from training/validation phase
with open("./results.txt", "r") as f:
    threshold = f.readline().strip()
    FIXED_THRESHOLD = float(threshold)

def calculate_fpr_fnr(scores: np.ndarray, true_labels: np.ndarray, threshold: float):
    """
    Calculate False Positive Rate (FPR) and False Negative Rate (FNR) for given scores and threshold.
    
    This function converts continuous scores to binary predictions using the provided threshold,
    then calculates various classification metrics including FPR and FNR.
    
    Args:
        scores (np.ndarray): Array of prediction scores/probabilities from the model
        true_labels (np.ndarray): Array of actual binary labels (0 or 1)
        threshold (float): Decision threshold for converting scores to binary predictions
        
    Returns:
        tuple: (fpr, fnr) - False Positive Rate and False Negative Rate
    """
    # Convert continuous scores to binary predictions using threshold
    # If score >= threshold, predict 1 (positive), otherwise predict 0 (negative)
    predictions = (scores >= threshold).astype(int)

    # Calculate confusion matrix components
    # TP: Model predicts positive (1) and actual is positive (1)
    tp = np.sum((true_labels == 1) & (predictions == 1))
    
    # TN: Model predicts negative (0) and actual is negative (0)
    tn = np.sum((true_labels == 0) & (predictions == 0))
    
    # FP: Model predicts positive (1) but actual is negative (0)
    fp = np.sum((true_labels == 0) & (predictions == 1))
    
    # FN: Model predicts negative (0) but actual is positive (1)
    fn = np.sum((true_labels == 1) & (predictions == 0))

    # Calculate FPR and FNR with division by zero protection
    # FPR = FP / (FP + TN) - Out of all actual negatives, how many were incorrectly classified as positive
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # FNR = FN / (FN + TP) - Out of all actual positives, how many were incorrectly classified as negative
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Display detailed metrics for the given threshold
    print(f"\n--- Metrics for Threshold = {threshold:.4f} ---")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"FPR (False Positive Rate): {fpr:.4f}")
    print(f"FNR (False Negative Rate): {fnr:.4f}")

    return fpr, fnr

if __name__ == "__main__":
    """
    Main execution block - loads SVM test results and calculates FPR/FNR metrics.
    
    This script expects:
    1. A CSV file with SVM scores in the first column and true labels in the second column
    2. A results.txt file containing the threshold value to use for classification
    """
    print(f"Loading results from: {INPUT_RESULTS_CSV}")
    try:
        # Load the CSV file containing test results
        # Expected format: first column = scores, second column = true labels
        data = pd.read_csv(INPUT_RESULTS_CSV, header=None)
        scores = data.iloc[:, 0].values
        true_labels = data.iloc[:, 1].values.astype(int)

        # Validate that data was loaded successfully
        if scores.size == 0 or true_labels.size == 0:
            print("Error: Loaded data is empty. Check the input CSV file.")
            exit()

        # Calculate and display FPR/FNR metrics using the predetermined threshold
        calculate_fpr_fnr(scores, true_labels, FIXED_THRESHOLD)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_RESULTS_CSV}'. Please ensure the path is correct and the SVM script has generated the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")