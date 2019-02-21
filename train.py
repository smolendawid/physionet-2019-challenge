import numpy as np
import pandas as pd
import logging
import os
import tqdm
from utils.path_utils import project_root

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
import datetime
import lightgbm as lgb
import matplotlib.pylab as plt
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


def save_features_importance(features_importance, features_names, path):
    features_importance = pd.DataFrame(sorted(zip(features_importance, features_names)),
                                       columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=features_importance.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(path)


def log(message: str='{}', value: any=None):
    print(message.format(value))
    logging.info(message.format(value))


def extract_features(training_files):
    features = []
    for training_file in training_files:
        example = pd.read_csv(training_file, sep='|')
        features = extract_example_features(example)
    return features


def extract_example_features(example):
    return features


def read_training_files(data_path):
    training_examples = []
    training_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    training_files.sort()

    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep='|')
        training_examples.append(example)

    return training_examples


if __name__ == '__main__':

    experiment_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(filename=os.path.join(project_root(), 'data', 'logs', experiment_time + '.log'),
                        level=logging.DEBUG)

    training_examples = read_training_files(os.path.join(project_root(), 'data', 'raw', 'training'))


    log(message='columns: {}', value=columns)

    skf = StratifiedKFold(n_splits=5)
    train_scores = []
    test_scores = []
    y_preds_test = []
    inds_test = []
    best_iters = []

    for i, (ind_train, ind_test) in enumerate(skf.split()):

        x_train = x_train_all[ind_train, :]
        x_test = x_train_all[ind_test, :]
        y_train = y_train_all[ind_train]
        y_test = y_train_all[ind_test]


        train_scores.append(mean_absolute_error(y_train, y_pred_train))
        test_scores.append(mean_absolute_error(y_test, y_pred_test))
        best_iters.append(num_iteration)
        y_preds_test.extend(list(y_pred_test))
        inds_test.extend(list(ind_test))
        log(message="Train score: {}", value=mean_absolute_error(y_train, y_pred_train))
        log(message="Test score: {}", value=mean_absolute_error(y_test, y_pred_test))

    log(message="\n\nMean train MAE: {}", value=np.mean(train_scores))
    log(message="Mean test MAE: {}", value=np.mean(test_scores))
    log(message="Std train MAE: {}", value=np.std(train_scores))
    log(message="Std test MAE: {}", value=np.std(test_scores))
    log(message="Best iter: {}\n\n", value=np.mean(best_iters))

    save_features_importance(model.feature_importances_, columns,
                             os.path.join(project_root(), 'data', 'plots', 'fi.png'))
