import numpy as np
import pandas as pd
import logging
import os
import datetime

from utils.path_utils import project_root

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from pytorch_classifier import PytorchClassifer
from compute_scores import compute_prediction_utility


def log(message: str='{}', value: any=None):
    print(message.format(value))
    logging.info(message.format(value))


if __name__ == '__main__':

    experiment_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(filename=os.path.join(project_root(), 'data', 'logs', experiment_time + '.log'),
                        level=logging.DEBUG)

    config = \
        {'epochs_num': 2,
         'batch_size': 4
         }
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', 'training_median.pickle'))
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

        x_train = [t for i, t in enumerate(training_examples) if i in ind_train]
        x_test = [t for i, t in enumerate(training_examples) if i in inds_test]

        batch_size = 4
        model = PytorchClassifer(config)
        model.fit(x_train, lengths_list)

        y_pred_train, y_train = model.predict(x_train)
        y_pred_test, y_test = model.predict(x_test)

        train_scores.append(compute_prediction_utility(y_train, y_pred_train))
        test_scores.append(compute_prediction_utility(y_test, y_pred_test))
        y_preds_test.extend(list(y_pred_test))
        inds_test.extend(list(ind_test))
        log(message="Train score: {}", value=compute_prediction_utility(y_train, y_pred_train))
        log(message="Test score: {}", value=compute_prediction_utility(y_test, y_pred_test))

    log(message="\n\nMean train MAE: {}", value=np.mean(train_scores))
    log(message="Mean test MAE: {}", value=np.mean(test_scores))
    log(message="Std train MAE: {}", value=np.std(train_scores))
    log(message="Std test MAE: {}", value=np.std(test_scores))
