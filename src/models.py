import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input, LSTM

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Statistical Algorithms
def get_total_average_predictions(X_train, y_train, X_test):
    average_value = y_train.mean()
    predictions = np.full(X_test.shape[0], average_value)
    return pd.Series(predictions, index=X_test.index)

def get_last_occurrence_predictions(X_train, y_train, X_test):
    default_value = y_train.mean()
    last_occurrence_results = {}
    for item_id in X_test['item_id']:
        X_train_lookup = X_train[X_train['item_id'] == item_id]
        if not X_train_lookup.empty:
            max_menu_week = X_train_lookup['menu_week'].max()
            last_occurrence_results[item_id] = y_train[(X_train['item_id'] == item_id) & (X_train['menu_week'] == max_menu_week)].values[0]
        else:
            last_occurrence_results[item_id] = default_value
    return X_test['item_id'].map(last_occurrence_results)

def get_average_occurrence_predictions(X_train, y_train, X_test):
    default_value = y_train.mean()
    avg_occurrence_results = {}
    for item_id in X_test['item_id']:
        X_train_lookup = X_train[X_train['item_id'] == item_id]
        if not X_train_lookup.empty:
            avg_occurrence_results[item_id] = y_train[X_train['item_id'] == item_id].mean()
        else:
            avg_occurrence_results[item_id] = default_value
    return X_test['item_id'].map(avg_occurrence_results)

# Linear Regression
def get_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# LightGBM
def get_lightgbm_model(X_train, y_train):
    params = {
        'verbose': -1,
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 100,
        'learning_rate': 0.2200044953698462,
        'max_depth': 18,
        'num_leaves': 669,
        'min_child_samples': 59,
        'min_split_gain': 1.4670533847983645e-06,
        'subsample': 0.7382953054809258,
        'colsample_bytree': 0.9648983195093089,
        'lambda_l1': 4.840165592010454e-07,
        'lambda_l2': 0.0025555820497953578,
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

# XGBoost
def get_xgboost_model(X_train, y_train):
    params = {
        'verbosity': 0,
        'objective': 'reg:logistic',
        'booster': 'gbtree',
        'eval_metric': 'mae',
        'n_estimators': 100,
        'learning_rate': 0.1759631363500291,
        'max_depth': 18,
        'min_child_weight': 2,
        'gamma': 0.0629948621294653,
        'subsample': 0.8156395766546685,
        'colsample_bytree': 0.5720051923599051,
        'colsample_bylevel': 0.8213911771088588,
        'lambda': 0.06733986476289976,
        'alpha': 0.8232226058749751,
        'random_state': 42,
        'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

# Random Forest
def get_random_forest_model(X_train, y_train):
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 4,
        'min_samples_leaf': 1,
        'max_features': 'log2',
        'max_leaf_nodes': None,
        'min_weight_fraction_leaf': 0.0,
        'max_samples': None,
        'criterion': 'friedman_mse',
        'min_impurity_decrease': 0.08,
        'random_state': 42,
        'n_jobs': -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

# RNN Model
def get_rnn_model(X_train, y_train, epochs=300, batch_size=16, units=64, time_steps=1):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Reshaping data for RNN: (samples, time_steps, features)
    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1]))

    # Build the RNN model
    model = Sequential()
    model.add(Input(shape=(time_steps, X_train_scaled.shape[1])))
    model.add(SimpleRNN(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and fit model
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler

# LSTM Model
def get_lstm_model(X_train, y_train, epochs=300, batch_size=16, units=64, time_steps=1):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Reshaping data for LSTM: (samples, time_steps, features)
    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_steps, X_train_scaled.shape[1])))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and fit model
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler

# Function to predict both neural network models and scale data
def nn_predict(trained_model, X_test, scaler, time_steps=1):
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], time_steps, X_test_scaled.shape[1]))

    y_pred_values = trained_model.predict(X_test_reshaped, verbose=0)
    return y_pred_values

# LightGBM Ranker
def get_lgbm_ranker_model(X_train, y_train):
    group_train = X_train.groupby('menu_week').size().to_list() 
    params = {
        'objective': 'lambdarank',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 13,
        'learning_rate': 0.06732857519006756,
        'n_estimators': 623,
        'min_child_weight': 6.207324113550803,
        'min_child_samples': 51,
        'subsample': 0.772260146512238,
        'subsample_freq': 8,
        'colsample_bytree': 0.6959641369551797,
        'reg_alpha': 0.0013516164018592264,
        'reg_lambda': 0.002763584077083565,
        'importance_type': 'split',
        'label_gain': list(range(max(group_train))),
        'verbose': -1
    }
    model = lgb.LGBMRanker(**params)
    model.fit(X_train, y_train, group=group_train)
    return model

# XGBoost Ranker
def get_xgboost_ranker_model(X_train, y_train):
    group_train = X_train.groupby('menu_week').size().to_list() 
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg',
        'learning_rate': 0.1413861549700997,
        'max_depth': 6,
        'min_child_weight': 17,
        'gamma': 1.0529385943580223,
        'subsample': 0.9283230260401305,
        'colsample_bytree': 0.6587909355240208,
        'colsample_bylevel': 0.5063142658549241,
        'lambda': 0.008776031157446995,
        'alpha': 0.23436060259654692,
        'random_state': 42,
        'n_jobs': -1,
        'ndcg_exp_gain': False,
        'random_state': 42,
    }

    num_boost_round = 458

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model

# CatBoost Ranker
def get_catboost_ranker_model(X_train, y_train):
    group_train = X_train['menu_week']
    
    params = {
        'learning_rate': 0.3477239622054686,
        'depth': 9,
        'iterations': 100,
        'loss_function': 'PairLogit',
        'verbose': False,
        'random_seed': 42,
        'l2_leaf_reg': 9.360490681305208,
        'colsample_bylevel': 0.7863055932866824,
        'subsample': 0.9448577817655559,
        'bootstrap_type': 'Bernoulli',
        'eval_metric': 'NDCG',
    }

    model = cb.CatBoostRanker(**params)
    model.fit(X_train, y_train, group_id=group_train)
    return model