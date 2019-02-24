
import pandas as pd

from joblib import Parallel, delayed


def _lag_feature(df, lags, on, col):
    tmp = df[on + [col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = on + [col+'_lag_'+str(i)]
        shifted['ICULOS'] += i
        df = pd.merge(df, shifted, on=on, how='left')
    return df


def _lag_existing_feature(df, lags, on, col):
    tmp = df[on + [col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = on + [col+'_lag_'+str(i)]
        shifted['ICULOS'] += i
        df = pd.merge(df, shifted, on=on, how='left')
    return df


def extract_example_features(df):
    return None


class FeatureGenerator(object):
    def __init__(self, training_examples_path, n_jobs=1):
        self.n_jobs = n_jobs
        self.training_examples_path = training_examples_path

    def read(self):
        training_examples = pd.read_pickle(self.training_examples_path)

        for counter, df in enumerate(training_examples[:10]):
            yield counter, df

    def features(self, seg_id, df):

        extracted_features = extract_example_features(df)

        return seg_id, extracted_features

    def generate(self):
        ids = []
        feature_list = []
        res = zip(Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.features)(i, df) for i, df in self.read()))
        # res.sort_values(by=['seg_id'])
        # res.drop('seg_id', axis=1, inplace=False)
        for i, r in res:
            ids.append(i)
            feature_list.append(r)
        return feature_list
