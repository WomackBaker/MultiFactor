from re import T
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


def train_svm(features, labels):
    """
    Trains an SVM classifier using the training dataset
    Sets the hyperparameters C and gamma before training

    Parameters
    ----------
        features: array_like
            The feature vector with a shape 43*N, where N is the size of training samples
        labels: array_like
            The true label vector with a shape 1*N
    """
    # uncomment the following statements to generate parameter values automatically
    # C_range = np.logspace(-1, 2, num=10)  # 10^-1 -- 10^2
    # gamma_range = np.logspace(-2, 1, num=10)  # 10^-2 -- 10^1

    # hard coded parameter values
    C_range = [1.0, 5.0, 10.0, 20.0, 40.0, 50]
    gamma_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['linear', 'rbf', 'poly'])

    # builds cross-validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

    # Finds the best parameters using cross validation, refer to the following link for more details
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)

    # trains the model using the training dataset
    grid.fit(features, labels)

    print("The best training parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    # return the trained model
    return grid


def train_knn(features, labels):
    # define the value range of parameter 'k'
    k_range = list(range(1, 21))
    param_grid = dict(n_neighbors=k_range, weights=['uniform', 'distance'])

    # perform cross-validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

    # adopt GridSearchCV for finding the best parameters
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    # train the model using the training dataset
    grid.fit(features, labels)

    print("The best training parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    return grid


def train_rf(features, labels):
    # define the value range of parameter 'number of trees'
    tree_range = list(range(50, 200, 10))
    param_grid = dict(n_estimators=tree_range)

    # perform cross-validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

    # adopt GridSearchCV for finding the best parameters
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, error_score='raise', n_jobs=-1)
    # train the model using the training dataset
    grid.fit(features, labels)

    print("The best training parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    return grid


def evaluate(model, features, labels):
    """
    Tests a trained model and plots the roc curve

    Parameters
    ----------
        model: sav
            Trained model in sav file
        features: array_like
            The feature vector with a shape 43*N, where N is the size of testing samples
        labels: array_like
            The true label vector with a shape 1*N
    """
    # test the model and save the predicted probability
    y_prob = model.predict_proba(features)
    y_predict = model.predict(features)

    # compute the fpr(far), tpr(tar), and eer
    fpr, tpr, thresholds = roc_curve(labels, y_prob[:, 1])

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print("EER %0.4f" % eer)
    thresh = interp1d(fpr, thresholds)(eer)
    print("THL %0.4f" % thresh)
    auc_result = auc(fpr, tpr)
    print("AUC %0.4f" % auc_result)

    return y_prob[:, 1], y_predict


# normalize train and test features, store the scaler
def normalize_train_test(train_features, test_features, save_scaler_to):
    features = np.append(train_features, test_features, 0)
    print(features)
    # features = normalize(features, 'max', axis=0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    pickle.dump(scaler, open(save_scaler_to, 'wb'))
    print(scaler.mean_)


# normalize the specific feature
def normalize_features(features, load_scaler_from):
    scaler = pickle.load(open(load_scaler_from, 'rb'))
    features = scaler.transform(features)
    return features


# train
def train(save_res_to, features, labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=train_svm):
    features = normalize_features(features, load_scaler_from)
    # print(features)
    if leave_one_subject_out:
        #===============original============
        save_res_to = './res/iter_user_test/' + video_type + '.csv'
        #==============resolution==================
        # save_res_to = './resolution/' + resolution + '/' + video_type + '.csv'
        for user in range(1, 11):
            print("user" + str(user))
            test_idx = input_data[input_data['user'] == user].index.tolist()
            # print(input_data)
            print(test_idx)
            training_idx = input_data[input_data['user'] != user].index.tolist()
            # print(test_idx)
            data = np.concatenate((features, labels), axis=1)
            training_feature = data[training_idx, :-1]
            training_label = data[training_idx, -1]
            test_feature = data[test_idx, :-1]
            test_label = data[test_idx, -1]
            # train a model with the training dataset
            model = method(features=training_feature, labels=training_label)
            y_score, y_predict = evaluate(model, features=test_feature, labels=test_label)
            y_score = pd.DataFrame(y_score, columns=["score"])
            test_label = pd.DataFrame(test_label, columns=["label"])
            user_id = pd.DataFrame([user] * len(test_label), columns=["user_id"])

            res = pd.concat([y_score, test_label], axis=1)
            res.to_csv(save_res_to, index=False, header=False, mode='a')  # append result to the csv file
    elif cross_validation:
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        index = 1
        for train_index, test_index in cv.split(features):
            save_res_to = './res/cross_validation/' + video_type + '_' + str(index) + '.csv'
            save_model_to_cv = './res/cross_validation/model_' + video_type + '_' + str(index) + '.sav'
            print(index)
            index += 1
            # print("Training Index", train_index)
            # print("Testing Index", test_index)
            data = np.concatenate((features, labels), axis=1)
            training_feature = data[train_index, :-1]
            training_label = data[train_index, -1]
            test_feature = data[test_index, :-1]
            test_label = data[test_index, -1]
            # training_feature, test_feature, training_label, test_label = features[train_index], features[test_index], labels[train_index], labels[test_index]
            model = method(features=training_feature, labels=training_label)
            y_score, y_predict = evaluate(model, features=test_feature, labels=test_label)
            y_score = pd.DataFrame(y_score, columns=["score"])
            test_label = pd.DataFrame(test_label, columns=["label"])
            # user_id = pd.DataFrame([user] * len(test_label), columns=["user_id"])
            res = pd.concat([y_score, test_label], axis=1)
            res.to_csv(save_res_to, index=False, header=False, mode='a')
            pickle.dump(model, open(save_model_to_cv, 'wb'))
    else:
        model = method(features=features, labels=np.ravel(labels, order='C'))
        pickle.dump(model, open(save_model_to, 'wb'))


# test
def test(features, labels, load_scaler_from, load_model_from, save_res_to):
    features = normalize_features(features, load_scaler_from)
    model = pickle.load(open(load_model_from, 'rb'))
    y_score, y_predict = evaluate(model, features=features, labels=labels)
    y_score = pd.DataFrame(y_score, columns=["score"])
    labels = pd.DataFrame(labels, columns=["label"])
    # res = pd.concat([y_score, labels], axis=1)
    res = pd.concat([y_score, labels], axis=1)
    print(res)
    res.to_csv(save_res_to, index=False, header=False)

if __name__ == '__main__':
    suffix = ''#'_zero'
    suffix_loo = '_27K'#''
    video_type = 'withpauseaccuracy'# suffix_loo.split('_')[1]  # knob, screen, button
    # print(video_type)
    is_normalized = False # if False, then need normalize
    train_model = False
    cross_validation = False
    leave_one_subject_out = False  # not used
    method = train_svm
    # mimic = True  # not used
    valid = False  # validate victim data to get threshold
    save_res_to = './res/iter_user_test/' + '.csv'
    save_scaler_to = './res/models/' + '_scaler.sav'
    save_model_to = './res/training_events/' + '_5.sav'  # uses 6 events
    # provides the correct directories where you saved the scaler and trained model
    load_scaler_from = save_scaler_to
    load_model_from = save_model_to

    # input_train_data = pd.read_csv('./timestamps_' + video_type + '_train' + suffix + '.csv', header=0)
    input_train_data = pd.read_csv('./train.csv',  header=None, delimiter=',')
    input_train_data = input_train_data.fillna(0)
    train_features = np.array(input_train_data.iloc[2:, 2:-1])
    train_labels = np.array(input_train_data.iloc[2:, -1]).reshape((-1, 1))

    # input_data = pd.read_csv('./test.csv', header=0)
    # input_data = input_data.fillna(0)
    # input_features = np.array(input_data.iloc[:, :-1])
    # input_labels = np.array(input_data.iloc[:, -1]).reshape((-1, 1))
    # print(input_labels)
    # if valid:
    #     input_valid_data = pd.read_csv('./data/features/mimic_victim_with_pauses/' + video_type + '.csv', header=None)
    #     input_valid_data = input_valid_data.fillna(0)
    #     valid_features = np.array(input_valid_data.iloc[:, 1:-1])
    #     valid_labels = np.array(input_valid_data.iloc[:, -1]).reshape((-1, 1))

    # input_test_data = pd.read_csv('./timestamps_' + video_type + '_test' + suffix + '.csv', header=0)
    input_test_data = pd.read_csv('./test.csv', header=None, delimiter=',')
    # input_test_data = pd.read_csv('./attack_with_pause/features_with_label.csv', header=0)
    input_test_data = input_test_data.fillna(0)
    print(input_train_data)
    test_features = np.array(input_test_data.iloc[2:, 2:-1])
    test_labels = np.array(input_test_data.iloc[2:, -1]).reshape((-1, 1))

    if not is_normalized:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        normalize_train_test(train_features, test_features, save_scaler_to)
        print("normalized!")
    elif train_model:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        #     train_labels = np.append(train_labels, valid_labels, 0)
        train(save_res_to, train_features, train_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    # elif valid:
    #     save_res_to = './res/mimic_test/attack_with_pauses/' + video_type + '.csv'
    #     test(valid_features, valid_labels, load_scaler_from, load_model_from, save_res_to)
    elif cross_validation:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    elif leave_one_subject_out:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    else:
        save_res_to = './test_result.csv'
        test(test_features, test_labels, load_scaler_from, load_model_from, save_res_to)
    
    suffix = ''#'_zero'
    suffix_loo = '_27K'#''
    video_type = 'withpauseaccuracy'# suffix_loo.split('_')[1]  # knob, screen, button
    # print(video_type)
    is_normalized = True # if False, then need normalize
    train_model = True
    cross_validation = False
    leave_one_subject_out = False  # not used
    #train_svm, train_knn, train_rf
    method = train_rf
    # mimic = True  # not used
    valid = False  # validate victim data to get threshold
    save_res_to = './res/iter_user_test/' + '.csv'
    save_scaler_to = './res/models/' + '_scaler.sav'
    save_model_to = './res/training_events/' + '_5.sav'  # uses 6 events
    # provides the correct directories where you saved the scaler and trained model
    load_scaler_from = save_scaler_to
    load_model_from = save_model_to

    # input_train_data = pd.read_csv('./timestamps_' + video_type + '_train' + suffix + '.csv', header=0)
    input_train_data = pd.read_csv('./train.csv',  header=None, delimiter=',')
    input_train_data = input_train_data.fillna(0)
    train_features = np.array(input_train_data.iloc[2:, 2:-1])
    train_labels = np.array(input_train_data.iloc[2:, -1]).reshape((-1, 1))

    # input_data = pd.read_csv('./test.csv', header=0)
    # input_data = input_data.fillna(0)
    # input_features = np.array(input_data.iloc[:, :-1])
    # input_labels = np.array(input_data.iloc[:, -1]).reshape((-1, 1))
    # print(input_labels)
    # if valid:
    #     input_valid_data = pd.read_csv('./data/features/mimic_victim_with_pauses/' + video_type + '.csv', header=None)
    #     input_valid_data = input_valid_data.fillna(0)
    #     valid_features = np.array(input_valid_data.iloc[:, 1:-1])
    #     valid_labels = np.array(input_valid_data.iloc[:, -1]).reshape((-1, 1))

    # input_test_data = pd.read_csv('./timestamps_' + video_type + '_test' + suffix + '.csv', header=0)
    input_test_data = pd.read_csv('./test.csv', header=None, delimiter=',')
    # input_test_data = pd.read_csv('./attack_with_pause/features_with_label.csv', header=0)
    input_test_data = input_test_data.fillna(0)
    print(input_train_data)
    test_features = np.array(input_test_data.iloc[2:, 2:-1])
    test_labels = np.array(input_test_data.iloc[2:, -1]).reshape((-1, 1))

    if not is_normalized:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        normalize_train_test(train_features, test_features, save_scaler_to)
        print("normalized!")
    elif train_model:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        #     train_labels = np.append(train_labels, valid_labels, 0)
        train(save_res_to, train_features, train_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    # elif valid:
    #     save_res_to = './res/mimic_test/attack_with_pauses/' + video_type + '.csv'
    #     test(valid_features, valid_labels, load_scaler_from, load_model_from, save_res_to)
    elif cross_validation:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    elif leave_one_subject_out:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    else:
        save_res_to = './test_result.csv'
        test(test_features, test_labels, load_scaler_from, load_model_from, save_res_to)
    
    suffix = ''#'_zero'
    suffix_loo = '_27K'#''
    video_type = 'withpauseaccuracy'# suffix_loo.split('_')[1]  # knob, screen, button
    # print(video_type)
    is_normalized = True # if False, then need normalize
    train_model = False
    cross_validation = False
    leave_one_subject_out = False  # not used
    method = train_svm
    # mimic = True  # not used
    valid = False  # validate victim data to get threshold
    save_res_to = './res/iter_user_test/' + '.csv'
    save_scaler_to = './res/models/' + '_scaler.sav'
    save_model_to = './res/training_events/' + '_5.sav'  # uses 6 events
    # provides the correct directories where you saved the scaler and trained model
    load_scaler_from = save_scaler_to
    load_model_from = save_model_to

    # input_train_data = pd.read_csv('./timestamps_' + video_type + '_train' + suffix + '.csv', header=0)
    input_train_data = pd.read_csv('./train.csv',  header=None, delimiter=',')
    input_train_data = input_train_data.fillna(0)
    train_features = np.array(input_train_data.iloc[2:, 2:-1])
    train_labels = np.array(input_train_data.iloc[2:, -1]).reshape((-1, 1))

    # input_data = pd.read_csv('./test.csv', header=0)
    # input_data = input_data.fillna(0)
    # input_features = np.array(input_data.iloc[:, :-1])
    # input_labels = np.array(input_data.iloc[:, -1]).reshape((-1, 1))
    # print(input_labels)
    # if valid:
    #     input_valid_data = pd.read_csv('./data/features/mimic_victim_with_pauses/' + video_type + '.csv', header=None)
    #     input_valid_data = input_valid_data.fillna(0)
    #     valid_features = np.array(input_valid_data.iloc[:, 1:-1])
    #     valid_labels = np.array(input_valid_data.iloc[:, -1]).reshape((-1, 1))

    # input_test_data = pd.read_csv('./timestamps_' + video_type + '_test' + suffix + '.csv', header=0)
    input_test_data = pd.read_csv('./test.csv', header=None, delimiter=',')
    # input_test_data = pd.read_csv('./attack_with_pause/features_with_label.csv', header=0)
    input_test_data = input_test_data.fillna(0)
    print(input_train_data)
    test_features = np.array(input_test_data.iloc[2:, 2:-1])
    test_labels = np.array(input_test_data.iloc[2:, -1]).reshape((-1, 1))

    if not is_normalized:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        normalize_train_test(train_features, test_features, save_scaler_to)
        print("normalized!")
    elif train_model:
        # if valid:
        #     train_features = np.append(train_features, valid_features, 0)
        #     train_labels = np.append(train_labels, valid_labels, 0)
        train(save_res_to, train_features, train_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    # elif valid:
    #     save_res_to = './res/mimic_test/attack_with_pauses/' + video_type + '.csv'
    #     test(valid_features, valid_labels, load_scaler_from, load_model_from, save_res_to)
    elif cross_validation:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    elif leave_one_subject_out:
        train(save_res_to, input_features, input_labels, load_scaler_from, save_model_to, leave_one_subject_out, cross_validation, method=method)
    else:
        save_res_to = './test_result.csv'
        test(test_features, test_labels, load_scaler_from, load_model_from, save_res_to)




# =================backup for one time training and testing================================
#     import random
#     random.seed(50)
#     random.shuffle(input_data)
#     split_1 = int(0.60 * len(input_data))
#
#     train_features = input_data[:split_1][:, :-1]
#     train_labels = input_data[:split_1][:, -1]
#
#     test_features = input_data[split_1:][:, :-1]
#     test_labels = input_data[split_1:][:, -1]
#
#     # train a model with the training dataset
#     model = train_svm(features=train_features, labels=train_labels)
#
#     if save_model_to:
#         pickle.dump(model, open(save_model_to, 'wb'))
#
#     y_score, y_predict = evaluate(model, features=test_features, labels=test_labels)
#
#     y_score = pd.DataFrame(y_score, columns=["score"])
#     test_labels = pd.DataFrame(test_labels, columns=["label"])
#
#     res = pd.concat([y_score, test_labels], axis=1, join_axes=[y_score.index])
#     res.to_csv(save_res_to, index=False, header=False)
