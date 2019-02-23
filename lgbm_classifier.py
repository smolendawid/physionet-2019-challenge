
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def save_features_importance(features_importance, features_names, path):
    features_importance = pd.DataFrame(sorted(zip(features_importance, features_names)),
                                       columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=features_importance.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(path)


class LGBMClassifier:
    def __init__(self, eval_set: list):
        self.eval_set = eval_set

        lgb_classifier_params = {'num_leaves': 60,
                                 'min_data_in_leaf': 120,
                                 'objective': 'binary',
                                 'max_depth': -1,
                                 'learning_rate': 0.001,
                                 "boosting": "gbdt",
                                 "metric": 'binary_error',
                                 "verbosity": -1,
                                 }
        self.model = lgb.LGBMClassifier(**lgb_classifier_params, n_estimators=20000, n_jobs=-1)

    def fit(self, x, y, lenghts):

        self.model.fit(x, y, sample_weight=None,
                       eval_set=self.eval_set,
                       verbose=50, early_stopping_rounds=500)

    def predict(self, x, num_iteration=None):
        if num_iteration is None:
            num_iteration = self.model.best_iteration_
        predictions = self.model.predict(x, num_iteration=num_iteration)

        return predictions
