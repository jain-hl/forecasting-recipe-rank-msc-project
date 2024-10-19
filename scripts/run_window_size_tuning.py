import matplotlib.pyplot as plt
import matplotlib

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_preprocessing import load_data, preprocess_data
from src.model_evaluation import run_evaluation_algos, run_evaluation

if __name__ == "__main__":
    matplotlib.use('Agg')

    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path)
    df = preprocess_data(df)

    training_window_range = range(100, 500, 50)
    algorithms = [
        {'name': 'Total Average', 'method': 'total_average'},
        {'name': 'Last Occurrence', 'method': 'last_occurrence'},
        {'name': 'Average Occurrence', 'method': 'average_occurrence'},
        {'name': 'Linear Regression', 'method': 'linear_regression'},
        {'name': 'LightGBM', 'method': 'lightgbm'},
        {'name': 'XGBoost', 'method': 'xgboost'},
        {'name': 'Random Forest', 'method': 'random_forest'},
        {'name': 'LightGBM Ranker', 'method': 'lgbm_ranker'},
        {'name': 'XGBoost Ranker', 'method': 'xgboost_ranker'},
        {'name': 'CatBoost Ranker', 'method': 'catboost_ranker'}
    ]

    # Dictionary to store the results for each algorithm
    mae_results = {algo['name']: [] for algo in algorithms}

    original_max_week = df['menu_week'].max() + 1

    # Run the evaluation
    total_week_iterations = 5
    weeks_before_target = 8
    target_week_range = range(original_max_week - total_week_iterations, original_max_week)

    for algo in algorithms:
        for training_window in training_window_range:
            if algo['method'] in ['total_average', 'last_occurrence', 'average_occurrence']:
                mean_mae = run_evaluation_algos(df, target_week_range, weeks_before_target, training_window, method=algo['method'])
            else:
                mean_mae = run_evaluation(df, target_week_range, weeks_before_target, training_window, method=algo['method'])
            print(f"Method: {algo['name']} | Training Window: {training_window} | Mean MAE = {mean_mae}", end = "\r")
            mae_results[algo['name']].append(mean_mae)

    matplotlib.use('TkAgg')
    if any(len(mae_values) > 0 for mae_values in mae_results.values()):
        # Plot the results if there is data
        plt.figure(figsize=(10, 6))

        for algo_name, mae_values in mae_results.items():
            if mae_values:
                plt.plot(training_window_range, mae_values, label=algo_name)

        plt.title('Mean MAE vs Training Window Size for Different Algorithms')
        plt.xlabel('Training Window Size')
        plt.ylabel('Mean MAE')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()