import lightgbm as lgbm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def load_model(path):
    return joblib.load(path)


def get_sepsis_score(values, thr):
    models_root = 'models/'
    models_paths = [os.path.join(models_root, m) for m in os.listdir(models_root) if m.endswith('.bin')]
    models_paths.sort()

    scores = np.zeros((len(values), ))
    for path in models_paths:
        model = load_model(path)
        scores += model.predict_proba(values)[:, 1]

    scores = scores / len(models_paths)
    labels = np.where(scores > thr, 1, 0)

    return (scores, labels)


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: %s input[.psv]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'
    (values, column_names) = read_challenge_data(input_file)

    # generate predictions
    thr = 0.29
    example = pd.DataFrame(values, columns=None, copy=True)

    example.ffill(inplace=True)
    example.bfill(inplace=True)
    example.fillna(0, inplace=True)

    (scores, labels) = get_sepsis_score(example.values, thr)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
