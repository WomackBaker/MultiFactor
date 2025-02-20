import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

def train_svm(features, labels):
    C_range = [1.0, 5.0, 10.0, 20.0, 40.0, 50]
    gamma_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['linear', 'rbf', 'poly'])
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print("SVM - Best params: %s, CV score=%.2f" % (grid.best_params_, grid.best_score_))
    return grid

def train_knn(features, labels):
    k_range = list(range(1, 21))
    param_grid = dict(n_neighbors=k_range, weights=['uniform', 'distance'])
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print("KNN - Best params: %s, CV score=%.2f" % (grid.best_params_, grid.best_score_))
    return grid

def train_rf(features, labels):
    tree_range = list(range(50, 200, 10))
    param_grid = dict(n_estimators=tree_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    grid.fit(features, labels)
    print("RF  - Best params: %s, CV score=%.2f" % (grid.best_params_, grid.best_score_))
    return grid

def evaluate(model, features, labels):
    y_prob = model.predict_proba(features)
    y_predict = model.predict(features)
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    auc_result = auc(fpr, tpr)
    print("EER = %.4f" % eer)
    print("Threshold @ EER = %.4f" % thresh)
    print("AUC = %.4f" % auc_result)
    return y_prob[:, 1], y_predict

def normalize_train_test(train_features, test_features, save_scaler_to):
    all_features = np.append(train_features, test_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    pickle.dump(scaler, open(save_scaler_to, 'wb'))
    print("Scaler mean: ", scaler.mean_)

def normalize_features(features, load_scaler_from):
    scaler = pickle.load(open(load_scaler_from, 'rb'))
    return scaler.transform(features)

def train(save_res_to, features, labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method):
    features = normalize_features(features, load_scaler_from)
    if leave_one_subject_out:
        pass
    elif cross_validation:
        pass
    else:
        model = method(features=features, labels=np.ravel(labels, order='C'))
        pickle.dump(model, open(save_model_to, 'wb'))

def test(features, labels, load_scaler_from, load_model_from, save_res_to):
    features = normalize_features(features, load_scaler_from)
    model = pickle.load(open(load_model_from, 'rb'))
    y_score, y_predict = evaluate(model, features, labels)
    y_score = pd.DataFrame(y_score, columns=["score"])
    labels_df = pd.DataFrame(labels, columns=["label"])
    res = pd.concat([y_score, labels_df], axis=1)
    res.to_csv(save_res_to, index=False, header=False)
    return res

if __name__ == '__main__':
    is_normalized = False
    train_model = True
    cross_validation = False
    leave_one_subject_out = False
    save_scaler_to = './res/models/_scaler.sav'
    model_filenames = {
        'svm': './res/training_events/svm_model.sav',
        'knn': './res/training_events/knn_model.sav',
        'rf':  './res/training_events/rf_model.sav'
    }
    save_result_files = {
        'svm': './test_result_svm.csv',
        'knn': './test_result_knn.csv',
        'rf':  './test_result_rf.csv'
    }
    input_train_data = pd.read_csv('./train.csv', header=None, delimiter=',')
    input_train_data = input_train_data.fillna(0)
    train_features = np.array(input_train_data.iloc[2:, 2:-1])
    train_labels = np.array(input_train_data.iloc[2:, -1]).reshape((-1, 1))
    input_test_data = pd.read_csv('./test.csv', header=None, delimiter=',')
    input_test_data = input_test_data.fillna(0)
    test_features = np.array(input_test_data.iloc[2:, 2:-1])
    test_labels = np.array(input_test_data.iloc[2:, -1]).reshape((-1, 1))
    if not is_normalized:
        print("Performing normalization and saving scaler...")
        normalize_train_test(train_features, test_features, save_scaler_to)
        is_normalized = True
    if train_model:
        for method, model_path in zip(
            [train_svm, train_knn, train_rf],
            [model_filenames['svm'], model_filenames['knn'], model_filenames['rf']]
        ):
            print("\n--- Training %s ---" % method.__name__)
            train('', train_features, train_labels, save_scaler_to, model_path, leave_one_subject_out, cross_validation, method)
    print("\n=== Testing all models ===")
    final_results = {}
    for method_name in ['svm', 'knn', 'rf']:
        print("\n--- Testing %s ---" % method_name.upper())
        res = test(test_features, test_labels, save_scaler_to, model_filenames[method_name], save_result_files[method_name])
        final_results[method_name] = res
    for name, df in final_results.items():
        print("*******************************************")
        print("Results for: ", name.upper())
        print(df.head())
