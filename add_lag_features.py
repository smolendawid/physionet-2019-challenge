import pandas as pd
import pickle
import os
import tqdm
import numpy as np
from utils.path_utils import project_root


def add_lag_features(training_examples):
    training_examples_lag = []
    for training_example in tqdm.tqdm(training_examples):
        lag_columns = [c + '_lag' for c in training_example.columns.values[:35]]

        lag_features = training_example.values[:-6, :35] - training_example.values[6:, :35]

        training_example = pd.concat([training_example, pd.DataFrame(columns=lag_columns)])
        training_example.loc[6:, lag_columns] = lag_features

        # training_example.ffill(inplace=True)
        # training_example.bfill(inplace=True)
        # training_example.fillna(0, inplace=True)

        training_examples_lag.append(training_example)

    return training_examples_lag


if __name__ == '__main__':

    data_name = 'training_filled.pickle'

    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_name))

    training_examples = add_lag_features(training_examples)

    with open(os.path.join(project_root(), 'data', 'processed', 'training_filled_lag.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)
