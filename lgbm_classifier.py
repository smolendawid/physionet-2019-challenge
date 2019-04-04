
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from compute_scores import normalized_utility_score
import pickle


def save_model(model, path):
    with open(path, 'wb') as fout:
        pickle.dump(model, fout)


def load_model(path):
    with open(path, 'rb') as fin:
        model = pickle.load(fin)

    return model


def save_features_importance(features_importance, features_names, path):
    features_importance = pd.DataFrame(sorted(zip(features_importance, features_names)),
                                       columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=features_importance.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(path)


lgb_classifier_params = {'num_leaves': 60,
                         'min_data_in_leaf': 120,
                         'objective': 'binary',
                         'max_depth': -1,
                         'learning_rate': 0.01,
                         # 'feature_fraction': 0.9,
                         # 'bagging_freq': 3,
                         # 'bagging_fraction': 0.9,
                         # 'bagging_seed': 0,
                         # 'feature_fraction_seed': 0,
                         'reg_alpha': 0,
                         'reg_lambda': 0,
                         'metric': 'auc',
                         'verbosity': -1,
                         'early_stopping_rounds': 500,
                         'scale_pos_weight': 20,
                         # 'is_unbalanced': False,
                         }


def eval_metric(labels, preds):
    labels = labels.reshape(1, -1)
    preds = np.where(preds > 0.5, 1, 0)
    preds = preds.reshape(1, -1)
    return 'normalized_utility_score', normalized_utility_score(labels, preds)[0], True


class LGBMClassifier:
    def __init__(self, config, writer, eval_set: list):
        self.eval_set = eval_set
        self.feature_importances_ = None

        self.model = lgb.LGBMClassifier(**lgb_classifier_params, n_estimators=20000, n_jobs=-1)

    def fit(self, examples, lenghts, sepsis):
        examples = pd.concat(examples)
        y_train = examples['SepsisLabel'].values
        x_train = examples.drop(['SepsisLabel'], axis=1).values

        eval_set = pd.concat(self.eval_set[0][0])
        y_eval = eval_set['SepsisLabel'].values
        x_eval = eval_set.drop(['SepsisLabel'], axis=1).values

        self.model.fit(x_train, y_train, sample_weight=None, eval_set=[(x_eval, y_eval)], verbose=50,)
                       # eval_metric=eval_metric)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, examples, num_iteration=None):
        references = []
        predictions = []
        for example in examples:
            reference = example['SepsisLabel'].values
            x = example.drop(['SepsisLabel'], axis=1).values

            if num_iteration is None:
                num_iteration = self.model.best_iteration_
            probas = self.model.predict_proba(x, num_iteration=num_iteration)[:, 1]

            search_result = threshold_search(reference, probas)
            thr = search_result['threshold']
            prediction = np.where(probas > thr, 1, 0)

            references.append(reference)
            predictions.append(prediction)

        return predictions, references


def threshold_search(y_true, y_proba):
    from compute_scores import normalized_utility_score
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.02 for i in range(100)]:
        score = normalized_utility_score(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result