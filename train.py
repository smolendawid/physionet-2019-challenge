import numpy as np
import pandas as pd
import logging
import os
import datetime
import shutil

from utils.path_utils import project_root

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from pytorch_classifier import PytorchClassifer
from compute_scores import normalized_utility_score
from config import nn_config


def log(message: str='{}', value: any=None):
    print(message.format(value))
    logging.info(message.format(value))


def setup_results(exp_time):

    log_root = os.path.join(project_root(), 'data', 'logs', exp_time)
    os.mkdir(log_root)
    logging.basicConfig(filename=os.path.join(log_root, exp_time + '.log'), level=logging.DEBUG)
    shutil.copy(os.path.join(project_root(), 'pytorch_classifier.py'), log_root)
    shutil.copy(os.path.join(project_root(), 'train.py'), log_root)


def get_split(ind_train, ind_test, training_examples, lengths_list, is_sepsis):

    x_train = [t for i, t in enumerate(training_examples) if i in ind_train]
    x_train_lens = [t for i, t in enumerate(lengths_list) if i in ind_train]
    is_sepsis_train = [t for i, t in enumerate(is_sepsis) if i in ind_train]
    x_test = [t for i, t in enumerate(training_examples) if i in ind_test]
    x_test_lens = [t for i, t in enumerate(lengths_list) if i in ind_test]
    is_sepsis_test = [t for i, t in enumerate(is_sepsis) if i in ind_test]

    return x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test


def main():
    exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    setup_results(exp_time)

    data_name = 'training_filled.pickle'
    log(message="Data name: {}", value=data_name)
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_name))
    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(l) for l in f.read().splitlines()]
    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(l) for l in f.read().splitlines()]

    skf = StratifiedKFold(n_splits=5)
    train_scores = []
    test_scores = []
    y_preds_test = []
    inds_test = []

    for i, (ind_train, ind_test) in enumerate(skf.split(training_examples, is_sepsis)):
        x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test = \
            get_split(ind_train, ind_test, training_examples, lengths_list, is_sepsis)

        model = PytorchClassifer(nn_config, os.path.join(project_root(), 'data', 'logs', exp_time))
        model.fit(x_train, x_train_lens, is_sepsis_train)
        y_pred_train, y_train = model.predict(x_train)
        y_pred_test, y_test = model.predict(x_test)

        test_score, _, _ = normalized_utility_score(y_test, y_pred_test)
        train_score, _, _ = normalized_utility_score(y_train, y_pred_train)

        test_scores.append(test_score)
        train_scores.append(train_score)
        y_preds_test.extend(y_pred_test)
        inds_test.extend(list(ind_test))
        log(message="Train score: {}", value=test_score)
        log(message="Test score: {}", value=test_score)

        # save_features_importance(model.feature_importances_, columns,
        #                          os.path.join(project_root(), 'data', 'plots', 'fi.png'))
        break

    log(message="\n\nMean train MAE: {}", value=np.mean(train_scores))
    log(message="Mean test MAE: {}", value=np.mean(test_scores))
    log(message="Std train MAE: {}", value=np.std(train_scores))
    log(message="Std test MAE: {}", value=np.std(test_scores))


if __name__ == '__main__':
    main()
