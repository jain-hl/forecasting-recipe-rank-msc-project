import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import normalize_data

def get_linear_regression_predictions(X_train, y_train, X_test):
    X_train, X_test = normalize_data(X_train, X_test)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

#XGBoost
def get_xgboost_predictions(X_train, y_train, X_test):
    params = {
        'objective': 'reg:logistic',
        'eval_metric': 'mae',
        'max_depth': 6,
    }
    model_xgboost = xgb.XGBRegressor(**params)
    model_xgboost.fit(X_train, y_train)
    predictions = model_xgboost.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

#LightGBM
def get_lightgbm_predictions(X_train, y_train, X_test):
    params = {
        'verbose': -1,
        'objective': 'regression',
        'metric': 'mae',
        'max_depth': 6,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

def get_random_forest_predictions(X_train, y_train, X_test):
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'criterion': 'friedman_mse',
        'random_state': 42,
        'n_jobs': -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

# LightGBM Ranker
def get_lgbm_ranker_predictions(X_train, y_train, X_test):
    group_train = X_train.groupby('menu_week').size().to_list() 
    params = {
        'objective': 'lambdarank',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'importance_type': 'split',
        'label_gain': list(range(max(group_train))),
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    }
    model = lgb.LGBMRanker(**params)
    model.fit(X_train, y_train, group=group_train)
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)

def get_xgboost_ranker_predictions(X_train, y_train, X_test):
    group_train = X_train.groupby('menu_week').size().to_list() 
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.15,
        'max_depth': 6,
        'min_child_weight': 17,
        'gamma': 1.0,
        'lambda': 0.01,
        'alpha': 0.2,
        'random_state': 42,
        'n_jobs': -1,
        'ndcg_exp_gain': False,  # Disable exponential gain for NDCG
        'random_state': 42,
    }

    num_boost_round = 100  # This replaces the 'n_estimators' parameter

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    return pd.Series(predictions, index=X_test.index)

def get_catboost_ranker_predictions(X_train, y_train, X_test):
    group_train = X_train['menu_week']
    params = {
        'learning_rate': 0.3,
        'iterations': 50,
        'loss_function': 'PairLogit',
        'verbose': False,
        'random_seed': 42,
        'bootstrap_type': 'Bernoulli',
        'eval_metric': 'NDCG',
        'random_seed': 42,
    }
    model = cb.CatBoostRanker(**params)
    model.fit(X_train, y_train, group_id=group_train)
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)