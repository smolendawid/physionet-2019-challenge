import numpy as np
import pandas as pd
import pickle
import os
import tqdm

from utils.path_utils import project_root

aggs = dict()
aggs_vital = {
    'HR': ['max', 'min', 'mean', 'var',],
    'O2Sat': ['max', 'min', 'mean', 'var',],
    'Temp': ['max', 'min', 'mean', 'var',],
    'SBP': ['max', 'min', 'mean', 'var',],
    'MAP': ['max', 'min', 'mean', 'var',],
    'DBP': ['max', 'min', 'mean', 'var',],
    'Resp': ['max', 'min', 'mean', 'var',],
    'EtCO2': ['max', 'min', 'mean', 'var',],
}

aggs_clinical = {
    'BaseExcess': ['max', 'min', 'mean', 'var',],
    'HCO3': ['max', 'min', 'mean', 'var',],
    'FiO2': ['max', 'min', 'mean', 'var',],
    'pH': ['max', 'min', 'mean', 'var',],
    'PaCO2': ['max', 'min', 'mean', 'var',],
    'SaO2': ['max', 'min', 'mean', 'var',],
    'AST': ['max', 'min', 'mean', 'var',],
    'BUN': ['max', 'min', 'mean', 'var',],
    'Alkalinephos': ['max', 'min', 'mean', 'var',],
    'Calcium': ['max', 'min', 'mean', 'var',],
    'Chloride': ['max', 'min', 'mean', 'var',],
    'Creatinine': ['max', 'min', 'mean', 'var',],
    'Bilirubin_direct': ['max', 'min', 'mean', 'var',],
    'Glucose': ['max', 'min', 'mean', 'var',],
    'Lactate': ['max', 'min', 'mean', 'var',],
    'Magnesium': ['max', 'min', 'mean', 'var',],
    'Phosphate': ['max', 'min', 'mean', 'var',],
    'Potassium': ['max', 'min', 'mean', 'var',],
    'Bilirubin_total': ['max', 'min', 'mean', 'var',],
    'TroponinI': ['max', 'min', 'mean', 'var',],
    'Hct': ['max', 'min', 'mean', 'var',],
    'Hgb': ['max', 'min', 'mean', 'var',],
    'PTT': ['max', 'min', 'mean', 'var',],
    'WBC': ['max', 'min', 'mean', 'var',],
    'Fibrinogen': ['max', 'min', 'mean', 'var',],
    'Platelets': ['max', 'min', 'mean', 'var',],
}

aggs.update(aggs_clinical)
aggs.update(aggs_vital)


def agg_feats(df):

    return None


def extract_features(df):
    df = agg_feats(df)
    return df


if __name__ == '__main__':
    data_name = 'training_concatenated.hdf'
    training_examples_path = os.path.join(project_root(), 'data', 'processed', data_name)
    df = pd.read_hdf(training_examples_path, key='df')
    df = df.iloc[:1000, :]

    df = extract_features(df)
    df.to_csv(os.path.join(project_root(), 'data', 'processed', 'training_features.csv'))

