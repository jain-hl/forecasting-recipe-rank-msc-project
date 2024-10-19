import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_absolute_error
import time

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models_pred_output import (
    get_linear_regression_predictions,
    get_xgboost_predictions,
    get_lightgbm_predictions,
    get_random_forest_predictions,
    get_lgbm_ranker_predictions,
    get_xgboost_ranker_predictions,
    get_catboost_ranker_predictions
)

def evaluate_predictions_live(all_errors, feature_errors, method, days_before_target=18, total_week_iterations=20):
    # Initialise error lists
    average_errors = []
    error_sems = []
    feature_maes = []
    feature_mae_sems = []

    for i in range(days_before_target + 7):
        errors_for_week = [iteration[i]['mean_absolute_error'] for iteration in all_errors if i < len(iteration)]
        average_error = np.mean(errors_for_week)
        error_sem = (np.std(errors_for_week) / np.sqrt(len(errors_for_week))) * 2  # 2 * standard error for 95% CI
        average_errors.append({'days_before': days_before_target - i, 'average_mean_absolute_error': average_error})
        error_sems.append({'days_before': days_before_target - i, 'error_sem': error_sem})

        # Process feature MAE
        feature_maes_for_week = [iteration[i]['feature_mae'] for iteration in feature_errors if i < len(iteration)]
        if feature_maes_for_week:
            feature_avg_mae = np.mean(feature_maes_for_week)
            feature_sem = (np.std(feature_maes_for_week) / np.sqrt(len(feature_maes_for_week))) * 2
            feature_maes.append({'days_before': days_before_target - i, 'feature_mean_absolute_error': feature_avg_mae})
            feature_mae_sems.append({'days_before': days_before_target - i, 'feature_sem': feature_sem})

    print(average_errors)

    average_errors_df = pd.DataFrame(average_errors)
    error_sems_df = pd.DataFrame(error_sems)
    feature_maes_df = pd.DataFrame(feature_maes)
    feature_mae_sems_df = pd.DataFrame(feature_mae_sems)

    # Create plot for short term model and feature uptake prediction
    plt.figure(figsize=(12, 6))
    plt.errorbar(feature_maes_df['days_before'], feature_maes_df['feature_mean_absolute_error'], 
                 yerr=feature_mae_sems_df['feature_sem'], fmt='--s', capsize=5, 
                 color='orange', ecolor='red', elinewidth=2, markerfacecolor='orange', label=f'Uptake Feature MAE (95% CI)')
    plt.errorbar(average_errors_df['days_before'], average_errors_df['average_mean_absolute_error'], 
                 yerr=error_sems_df['error_sem'], fmt='-o', capsize=5, 
                 color='purple', ecolor='blue', elinewidth=2, markerfacecolor='purple', label='Predicted MAE (95% CI)')
    plt.ylim(-0.005, 0.165)
    plt.yticks(np.arange(0, 0.17, 0.01))
    plt.xticks(np.arange(days_before_target, -7, -1))
    plt.xlim(-7, days_before_target+1)
    plt.xlabel('Days Before Target')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'{method}: Average Mean Absolute Error over {days_before_target} Days Before Target Across {total_week_iterations} Iterations')
    plt.gca().invert_xaxis()
    plt.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.4)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.grid(True)
    plt.legend()
    plt.show()

# Function to run evaluation given algorithm
def run_evaluation_live(df, target_week_range, days_before_target=18, method='linear_regression'):
    df = df.copy()

    total_week_iterations = len(target_week_range)
    all_errors = []
    feature_errors = []
    num_models = 0
    total_models = total_week_iterations * (days_before_target + 7)

    method_dict = {
    'linear_regression': get_linear_regression_predictions,
    'xgboost': get_xgboost_predictions,
    'lightgbm': get_lightgbm_predictions,
    'random_forest': get_random_forest_predictions,
    'lgbm_ranker': get_lgbm_ranker_predictions,
    'xgboost_ranker': get_xgboost_ranker_predictions,
    'catboost_ranker': get_catboost_ranker_predictions,
    }

    
    if method not in method_dict:
        raise ValueError("Invalid method specified.")

    if (method != 'lgbm_ranker'):
        df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=False)
    

    for target_week in target_week_range:
        iteration_errors = []
        feature_iteration_errors = []
        elapsed_times = []
        for days_before in range(0, days_before_target + 7):
            start_time = time.time()
            num_models += 1
            # Add the uptake_at_lead_day column for the current days_before
            columns_to_keep = [f'uptake_at_lead_day_{days_before_target + 1 - days_before}']
            for i in range(0, days_before + 1):
                columns_to_keep.append(f'diff_uptake_at_lead_day_{days_before_target + 1 - i}')
                
            # columns_to_keep = [f'uptake_at_lead_day_{days_before_target + 1 - days_before}', f'diff_uptake_at_lead_day_{days_before_target + 1 - days_before}']
            columns_to_remove = [col for col in df.columns if 'uptake_at_lead_day_' in col and col not in columns_to_keep]
            df_reduced = df.drop(columns=columns_to_remove)
            # columns = df_reduced.filter(like="uptake_at_lead_day").columns
            
            buffer = (max(0, days_before_target + 1 - days_before) // 7) + 1

            train_df = df_reduced[df_reduced['menu_week'] <= target_week - buffer]
            test_df = df_reduced[df_reduced['menu_week'] == target_week]

            X_train = train_df.drop(columns=['recipe_rank'])
            y_train = train_df['recipe_rank']

            X_test = test_df.drop(columns=['recipe_rank'])
            y_test = test_df['recipe_rank']

            y_pred_values = method_dict[method](X_train, y_train, X_test)
            y_pred_values = pd.Series(y_pred_values).rank(ascending=True, pct=True, method='first').to_numpy()
            if (method == 'lgbm_ranker'):
                y_test = pd.Series(y_test).rank(ascending=False, pct=True, method='first')
                y_pred_values = 1 - y_pred_values
            mean_absolute_error_value = mean_absolute_error(y_test.values, y_pred_values)
            iteration_errors.append({'days_before': days_before_target - days_before, 'mean_absolute_error': mean_absolute_error_value})

            uptake_feature_col = f'uptake_at_lead_day_{days_before_target + 1 - days_before}'
            if uptake_feature_col in X_test.columns:
                y_feature_test = X_test[uptake_feature_col].values
                feature_mae_value = mean_absolute_error(y_test, y_feature_test)
                feature_iteration_errors.append({'days_before': days_before_target - days_before, 'feature_mae': feature_mae_value})
            else:
                feature_iteration_errors.append({'days_before': days_before_target - days_before, 'feature_mae': mean_absolute_error_value})
                
            time_elapsed = time.time() - start_time
            elapsed_times.append(time_elapsed)
            if len(elapsed_times) > 20:
                elapsed_times.pop(0)

            if elapsed_times:
                avg_time_elapsed = sum(elapsed_times) / len(elapsed_times)
                minutes, seconds = divmod(avg_time_elapsed * (total_models - num_models), 60)

            print(f"No. of models progress: {num_models}/{total_models} | Estimated Time Left: {int(minutes)} minutes and {int(seconds)} seconds              ", end='\r')

        all_errors.append(iteration_errors)
        feature_errors.append(feature_iteration_errors)
        
    evaluate_predictions_live(all_errors, feature_errors, method, days_before_target, total_week_iterations)