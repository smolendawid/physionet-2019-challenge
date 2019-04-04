import lightgbm as lgbm
import os
import sys
import pickle
import numpy as np


def load_model(path):
    with open(path, 'rb') as fin:
        model = pickle.load(fin)

    return model


def get_sepsis_score(values):
    models_paths = [m for m in os.listdir('./') if m.endswith('.pkl')]

    probas = np.zeros((len(values), ))
    for path in models_paths:
        model = load_model(path)
        probas += model.predict_proba(values)[:, 1]

    labels = np.where(probas/len(models_paths) > 0.5, 1, 0)

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
    (scores, labels) = get_sepsis_score(values)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
