nn_config = \
    {'epochs_num': 20,
     'batch_size': 1,
     'input_size': 39,
     'hidden_size': 39,
     'num_of_heads': 3,
     'num_layers': 4,
     'dropout': 0.0,
     'lr': 0.0001,
     'size_average': True,
     'clipping': 50,
     'to_concat': True,
     }


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
                         'early_stopping_rounds': 100,
                         'scale_pos_weight': 20,
                         # 'is_unbalanced': False,
                         }


