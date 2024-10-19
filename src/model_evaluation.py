import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from matplotlib.ticker import ScalarFormatter
import time
import xgboost as xgb

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import (
    get_total_average_predictions, 
    get_last_occurrence_predictions, 
    get_average_occurrence_predictions, 
    get_linear_regression_model, 
    get_lightgbm_model, 
    get_xgboost_model, 
    get_random_forest_model, 
    get_rnn_model, 
    get_lstm_model, 
    nn_predict, 
    get_lgbm_ranker_model, 
    get_xgboost_ranker_model, 
    get_catboost_ranker_model
)

def evaluate_predictions_algos(all_errors, method, weeks_before_target=8, total_week_iterations=20):
    min_y_range = 0.05
    average_errors, lower_bound_errors, upper_bound_errors = [], [], []

    for i in range(weeks_before_target):
        errors_for_week = [iteration[i]['mean_absolute_error'] for iteration in all_errors if i < len(iteration)]
        if errors_for_week:
            average_error = np.mean(errors_for_week)
            sem = np.std(errors_for_week) / np.sqrt(len(errors_for_week))
            ci = 1.96 * sem  # 95% confidence interval

            average_errors.append({'week_before_target': weeks_before_target - i, 'average_mean_absolute_error': average_error})
            lower_bound_errors.append(average_error - ci)
            upper_bound_errors.append(average_error + ci)

    # Convert to DataFrame
    average_errors_df = pd.DataFrame(average_errors)
    avg_errors_array = average_errors_df['average_mean_absolute_error'].to_numpy()

    # Calculate y-limits and add margins if necessary
    y_min, y_max = min(lower_bound_errors), max(upper_bound_errors)
    y_range = y_max - y_min
    if y_range < min_y_range:
        margin = min_y_range * 0.5
        y_min = np.mean([y_min, y_max]) - margin
        y_max = np.mean([y_min, y_max]) + margin
    else:
        margin = y_range * 0.1
        y_min -= margin
        y_max += margin

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(average_errors_df['week_before_target'], avg_errors_array, marker='o', color='b', linestyle='-', label='Average MAE', lw=2, markersize=6)
    plt.fill_between(average_errors_df['week_before_target'], lower_bound_errors, upper_bound_errors, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.ylim([y_min, y_max])
    plt.xlabel('Weeks Before Target', fontsize=14)
    plt.ylabel('Average Mean Absolute Error', fontsize=14)
    plt.title(f'{method}: Average MAE over {weeks_before_target} Weeks Before Target Across {total_week_iterations} Iterations', fontsize=16)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


def evaluate_predictions(all_errors, method, weeks_before_target=8, total_week_iterations=20):
    min_y_range = 0.05
    average_errors, lower_bound_errors, upper_bound_errors = [], [], []

    for i in range(1, weeks_before_target + 1):
        errors_for_week = all_errors.get(i, [])
        if errors_for_week:
            average_error = np.mean(errors_for_week)
            sem = np.std(errors_for_week) / np.sqrt(len(errors_for_week))
            ci = 1.96 * sem  # 95% confidence interval

            average_errors.append({'week_before_target': i, 'average_mean_absolute_error': average_error})
            lower_bound_errors.append(average_error - ci)
            upper_bound_errors.append(average_error + ci)
    
    # Convert to DataFrame
    average_errors_df = pd.DataFrame(average_errors)
    avg_errors_array = average_errors_df['average_mean_absolute_error'].to_numpy()

    # Calculate y-limits and add margins if necessary
    y_min, y_max = min(lower_bound_errors), max(upper_bound_errors)
    y_range = y_max - y_min
    if y_range < min_y_range:
        margin = min_y_range * 0.5
        y_min = np.mean([y_min, y_max]) - margin
        y_max = np.mean([y_min, y_max]) + margin
    else:
        margin = y_range * 0.1
        y_min -= margin
        y_max += margin

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(average_errors_df['week_before_target'], avg_errors_array, marker='o', color='b', linestyle='-', label='Average MAE', lw=2, markersize=6)
    plt.fill_between(average_errors_df['week_before_target'], lower_bound_errors, upper_bound_errors, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.ylim([y_min, y_max])
    plt.xlabel('Weeks Before Target', fontsize=14)
    plt.ylabel('Average Mean Absolute Error', fontsize=14)
    plt.title(f'{method}: Average MAE over {weeks_before_target} Weeks Before Target Across {total_week_iterations} Iterations', fontsize=16)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


def run_evaluation_algos(df, target_week_range, weeks_before_target, training_window=None, method='average_occurrence'):
    df = df.copy()
    total_week_iterations = len(target_week_range)
    all_errors = []
    num_models = 0
    training_window = max(target_week_range) + 1 if training_window is None else training_window
    
    method_dict = {
        'total_average': get_total_average_predictions,
        'last_occurrence': get_last_occurrence_predictions,
        'average_occurrence': get_average_occurrence_predictions,
    }

    df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)

    for max_week in target_week_range:
        menu_week_range = range(max_week - weeks_before_target, max_week)
        iteration_errors = []
        
        for menu_week_index in menu_week_range:
            start_time = time.time()
            num_models += 1

            train_df = df[(df['menu_week'] >= max(menu_week_index - training_window, 0)) & (df['menu_week'] <= menu_week_index)]
            test_df = df[df['menu_week'] == max_week]

            X_train = train_df.drop(columns=['recipe_rank'])
            y_train = train_df['recipe_rank']

            X_test = test_df.drop(columns=['recipe_rank'])
            y_test = test_df['recipe_rank']
            
            y_pred_values = method_dict[method](X_train, y_train, X_test)
            y_pred_values = pd.Series(y_pred_values).rank(ascending=True, method='average') / len(y_pred_values)

            mean_absolute_error_value = mean_absolute_error(y_test.values, y_pred_values.values)
            iteration_errors.append({'menu_week': menu_week_index, 'mean_absolute_error': mean_absolute_error_value})

            current_mean_error = np.mean([error['mean_absolute_error'] for errors in all_errors + [iteration_errors] for error in errors])

            time_taken = time.time() - start_time
            print(f"No. of models progress: {num_models}/{weeks_before_target * total_week_iterations} | Rolling MAE: {current_mean_error:.4f} | Estimated Time Left: {(1/60 * time_taken * (weeks_before_target * total_week_iterations - num_models)):.1f} minutes", end='\r', flush=True)

        all_errors.append(iteration_errors)
        
    evaluate_predictions_algos(all_errors, method, weeks_before_target, total_week_iterations)
    return current_mean_error


def run_evaluation(df, target_week_range, weeks_before_target, training_window=None, method='xgboost'):
    df = df.copy()
    total_week_iterations = len(target_week_range)
    all_errors = {weeks_before: [] for weeks_before in range(1, weeks_before_target + 1)}
    training_window = max(target_week_range) + 1 if training_window is None else training_window

    num_models = 0
    num_inferences = 0
    total_models = weeks_before_target + total_week_iterations - 1
    total_inferences = weeks_before_target * total_week_iterations

    method_dict = {
        'linear_regression': get_linear_regression_model,
        'lightgbm': get_lightgbm_model,
        'xgboost': get_xgboost_model,
        'random_forest': get_random_forest_model,
        'lgbm_ranker': get_lgbm_ranker_model,
        'xgboost_ranker': get_xgboost_ranker_model,
        'catboost_ranker': get_catboost_ranker_model,
        'rnn': get_rnn_model,
        'lstm': get_lstm_model,
    }

    rank_percentile_models = ['linear_regression', 'lightgbm', 'xgboost', 'random_forest', 'catboost_ranker', 'rnn', 'lstm']

    if method in rank_percentile_models:
        df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)

    model_func = method_dict[method]
    
    for menu_week_index in range(min(target_week_range) - weeks_before_target, max(target_week_range)):
        num_models += 1
        start_time = time.time()

        train_df = df[(df['menu_week'] >= max(menu_week_index - training_window, 0)) & (df['menu_week'] <= menu_week_index)]
        X_train = train_df.drop(columns=['recipe_rank'])
        y_train = train_df['recipe_rank']

        if method in ['rnn', 'lstm']:
            trained_model, scaler = model_func(X_train, y_train)
        else: 
            trained_model = model_func(X_train, y_train)

        time_taken = time.time() - start_time

        for max_week in target_week_range:
            if max_week > menu_week_index >= max_week - weeks_before_target:
                num_inferences += 1

                test_df = df[df['menu_week'] == max_week]
                X_test = test_df.drop(columns=['recipe_rank'])
                y_test = test_df['recipe_rank']

                if method == 'xgboost_ranker':
                    dtest = xgb.DMatrix(X_test)
                    y_pred_values = trained_model.predict(dtest)
                elif method in ['rnn', 'lstm']:
                    y_pred_values = nn_predict(trained_model, X_test, scaler)
                    y_pred_values = y_pred_values.flatten()
                else:
                    y_pred_values = trained_model.predict(X_test)

                y_pred_values = pd.Series(y_pred_values).rank(ascending=True, pct=True, method='first')

                if method not in rank_percentile_models:
                    y_test = pd.Series(y_test).rank(ascending=True, pct=True, method='first')

                mean_absolute_error_value = mean_absolute_error(y_test.values, y_pred_values.values)
                weeks_before = max_week - menu_week_index
                all_errors[weeks_before].append(mean_absolute_error_value)
                current_mean_error = np.mean([error for errors in all_errors.values() for error in errors])

                print(f"No. of models progress: {num_models}/{total_models} | No. of inferences progress: {num_inferences}/{total_inferences} | Rolling MAE: {current_mean_error:.4f} | Estimated Time Left: {(1/60 * time_taken * (total_models - num_models)):.1f} minutes", end='\r')

    evaluate_predictions(all_errors, method, weeks_before_target, total_week_iterations)
    return current_mean_error