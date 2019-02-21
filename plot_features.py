import pandas as pd
import numpy as np
import os
import tqdm
from utils.path_utils import project_root
import seaborn as sns
import matplotlib.pylab as plt
from train import read_training_files
import pickle


# for column in x_train_all:
#     plt.figure(figsize=(13, 7))
#     plt.title(column)
#     plt.ylabel('Value')
#     plt.xlabel('Bin')
#     sns.violinplot(x='binned_y', y=column, data=x_train_all)
#     plt.savefig(os.path.join(project_root(), 'data', 'plots', 'features_vs_bins', f'{column}.png'))
#     plt.close()

def plot_sepsis_label(training_examples):

    for training_example in tqdm.tqdm(training_examples):

        training_example.fillna(0, inplace=True)
        if 1 in training_example['SepsisLabel'].values:
            plt.figure()
            plt.plot(training_example['SepsisLabel'], c="g")
            plt.show(block=False)
    print()


def plot_length_hist(training_examples):

    lengths = []
    for training_example in tqdm.tqdm(training_examples):
        lengths.append(len(training_example['SepsisLabel'].values))

    plt.figure()
    ax = sns.distplot(lengths)
    print()


def calculate_sepsis():
    # occ = 0
    # for training_example in tqdm.tqdm(training_examples):
    #     if 1 in training_example['SepsisLabel'].values:
    #         occ += 1
    # print(occ)

    data_path = os.path.join(project_root(), 'data', 'processed', 'training')

    training_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    training_files.sort()
    for training_file in tqdm.tqdm(training_files):
        training_example = pd.read_csv(training_file, sep='|')
        if 1 in training_example['SepsisLabel'].values:
            print(training_file)


def plot_start_sepsis_hist(training_examples):
    start_inds = []
    rel_start_inds = []
    for training_example in tqdm.tqdm(training_examples):
        training_example.fillna(0, inplace=True)
        occurances = np.where(training_example['SepsisLabel'].values == 1.)[0]
        if len(occurances) == 0:
            continue
            start_inds.append(-1)
            rel_start_inds.append(-1)
        else:
            start_inds.append(occurances[0])
            rel_start_inds.append(float(occurances[0])/len(training_example['SepsisLabel'].values))

    plt.figure()
    ax = sns.distplot(start_inds)

    plt.figure()
    ax = sns.distplot(rel_start_inds)
    print()


def check_if_exists_empty_after_non_empty(training_examples):
    occured = False
    for training_example in tqdm.tqdm(training_examples):

        training_example.fillna(0, inplace=True)
        if 1 in training_example['SepsisLabel'].values:
            is_sepsis = False
            for i in training_example['SepsisLabel'].values:
                if i == 1:
                    is_sepsis = True
                if is_sepsis and i == 0:

                    plt.figure()
                    plt.plot(training_example['SepsisLabel'], c="g")
                    plt.show(block=False)
                    occured = True
    if occured:
        print('Occured')


if __name__ == '__main__':

    with open(os.path.join(project_root(), 'data', 'processed', 'training.pickle'), 'rb') as fp:
        training_examples = pickle.load(fp)

    # plot_sepsis_label(training_examples)
    # plot_length_hist(training_examples)
    # check_if_exists_empty_after_non_empty(training_examples)
    calculate_sepsis()
    plot_start_sepsis_hist(training_examples)
