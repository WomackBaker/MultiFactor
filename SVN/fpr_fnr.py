import pandas as pd
import numpy as np

INPUT_RESULTS_CSV = "results/test_result_svm.csv"
FIXED_THRESHOLD = 0.5

def calculate_fpr_fnr(scores: np.ndarray, true_labels: np.ndarray, threshold: float):
    predictions = (scores >= threshold).astype(int)

    tp = np.sum((true_labels == 1) & (predictions == 1))
    tn = np.sum((true_labels == 0) & (predictions == 0))
    fp = np.sum((true_labels == 0) & (predictions == 1))
    fn = np.sum((true_labels == 1) & (predictions == 0))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n--- Metrics for Threshold = {threshold:.4f} ---")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"FPR (False Positive Rate): {fpr:.4f}")
    print(f"FNR (False Negative Rate): {fnr:.4f}")

    return fpr, fnr

if __name__ == "__main__":
    print(f"Loading results from: {INPUT_RESULTS_CSV}")
    try:
        data = pd.read_csv(INPUT_RESULTS_CSV, header=None)
        scores = data.iloc[:, 0].values
        true_labels = data.iloc[:, 1].values.astype(int)

        if scores.size == 0 or true_labels.size == 0:
            print("Error: Loaded data is empty. Check the input CSV file.")
            exit()

        calculate_fpr_fnr(scores, true_labels, FIXED_THRESHOLD)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_RESULTS_CSV}'. Please ensure the path is correct and the SVM script has generated the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")