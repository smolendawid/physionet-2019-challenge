import pandas as pd
import numpy as np
import os
import pickle
import tqdm
from utils.path_utils import project_root


def write_pickle(training_files):

    lengths = []
    all_data = pd.DataFrame()
    training_examples = []
    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep=',')
        lengths.append(len(example))
        all_data = pd.concat([all_data, example])

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt', 'w')) as f:
        [f.write('{}\n'.format(l)) for l in lengths]

    with open(os.path.join(project_root(), 'data', 'processed', 'training_raw.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)

    medians = all_data.median(axis=0, skipna=True)
    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep=',')
        example.fillna(medians, inplace=True)
        training_examples.append(example)

    with open(os.path.join(project_root(), 'data', 'processed', 'training_median.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)

    means = all_data.mean(axis=0, skipna=True)
    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep=',')
        example.fillna(means, inplace=True)
        training_examples.append(example)

    with open(os.path.join(project_root(), 'data', 'processed', 'training_mean.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)

    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep=',')
        example.fillna(0, inplace=True)
        training_examples.append(example)

    with open(os.path.join(project_root(), 'data', 'processed', 'training_zeros.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)


if __name__ == '__main__':

    data_path = os.path.join(project_root(), 'data', 'processed', 'training')

    training_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    training_files.sort()

    write_pickle(training_files)
