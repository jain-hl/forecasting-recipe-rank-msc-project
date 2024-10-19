import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import optuna
import optuna.visualization as vis

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.utils import custom_time_series_cv
from src.data_preprocessing import load_data, preprocess_data

# Function to tune LightGBM Hyperparameters
def tune_lightgbm_hyperparameters(df, start_week, n_splits=3, n_trials=150):
    df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)
    initial_data = df[df['menu_week'] < start_week]
    
    X = initial_data.drop(columns=['recipe_rank'])
    y = initial_data['recipe_rank']

    def objective(trial):
        param = {
            'verbosity': -1,
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 100,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'num_leaves': trial.suggest_int('num_leaves', 20, 800),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
        }

        maes = []
        tscv = custom_time_series_cv(X, n_splits)
        for train_index, valid_index in tscv:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # LightGBM model
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, preds)
            maes.append(mae)
        
        return np.mean(maes)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()

    return study.best_params

# Function to tune XGBoost Hyperparameters
def tune_xgboost_hyperparameters(df, start_week, n_splits=3, n_trials=150):
    df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)
    initial_data = df[df['menu_week'] < start_week]
    
    X = initial_data.drop(columns=['recipe_rank'])
    y = initial_data['recipe_rank']

    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:logistic',
            'booster': 'gbtree',
            'eval_metric': 'mae',
            'n_estimators': 100,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
        }

        maes = []
        tscv = custom_time_series_cv(X, n_splits)
        for train_index, valid_index in tscv:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, preds)
            maes.append(mae)
        return np.mean(maes)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()
    
    return study.best_params

# Function to tune Random Forest Hyperparameters
def tune_random_forest_hyperparameters(df, start_week, n_splits=3, n_trials=150):
    df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)
    initial_data = df[df['menu_week'] < start_week]
    
    X = initial_data.drop(columns=['recipe_rank'])
    y = initial_data['recipe_rank']

    def objective(trial):
        param = {
            'n_estimators': 100,
            'max_depth': trial.suggest_int('max_depth', 5, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000) if trial.suggest_categorical('use_max_leaf_nodes', [True, False]) else None,
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 1.0),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.suggest_categorical('use_max_samples', [True, False]) else None,
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
            'random_state': 42,
            'n_jobs': -1,
        }

        maes = []
        tscv = custom_time_series_cv(X, n_splits)
        for train_index, valid_index in tscv:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            model = RandomForestRegressor(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            preds = pd.Series(preds).rank(ascending=True, method='first') / len(preds)
            mae = mean_absolute_error(y_valid, preds)
            maes.append(mae)
        return np.mean(maes)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()
    
    return study.best_params

if __name__ == "__main__":

    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Hyperparameter Tuning
    original_max_week = df['menu_week'].max() + 1

    # Run the evaluation
    total_week_iterations = 52
    weeks_before_target = 8
    target_week_range = range(original_max_week - total_week_iterations, original_max_week)

    # Tuning
    n_splits = 3
    n_trials = 3

    start_week_for_tuning = original_max_week - total_week_iterations - weeks_before_target - n_splits

    best_params_lightgbm = tune_lightgbm_hyperparameters(df, start_week_for_tuning, n_splits=n_splits, n_trials=n_trials); print(best_params_lightgbm)
    best_params_xgboost = tune_xgboost_hyperparameters(df, start_week_for_tuning, n_splits=n_splits, n_trials=n_trials); print(best_params_xgboost)
    best_params_random_forest = tune_random_forest_hyperparameters(df, start_week_for_tuning, n_splits=n_splits, n_trials=n_trials); print(best_params_random_forest)