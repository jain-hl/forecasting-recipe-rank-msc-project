import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.data_preprocessing import load_data, preprocess_data
from src.model_evaluation import run_evaluation_algos, run_evaluation

if __name__ == "__main__":

    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Model Execution
    setback = 0
    original_max_week = df['menu_week'].max() + 1 - setback

    total_week_iterations = 52
    weeks_before_target = 8
    target_week_range = range(original_max_week - total_week_iterations, original_max_week)

    # print("Total Average Algorithm:"); run_evaluation_algos(df, target_week_range, weeks_before_target, training_window=None, method='total_average'); print("\n")

    # print("Last Occurrence Algorithm:"); run_evaluation_algos(df, target_week_range, weeks_before_target, training_window=None, method='last_occurrence'); print("\n")

    # print("Average Occurrence Algorithm:"); run_evaluation_algos(df, target_week_range, weeks_before_target, training_window=120, method='average_occurrence'); print("\n")

    # print("Linear Regression:"); run_evaluation(df, target_week_range, weeks_before_target, training_window=50, method='linear_regression'); print("\n")

    # print("LightGBM:"); run_evaluation(df, target_week_range, weeks_before_target, training_window=100, method='lightgbm'); print("\n")

    print("XGBoost:"); run_evaluation(df, target_week_range, weeks_before_target, training_window=300, method='xgboost'); print("\n")

    # print("Random Forest:"); run_evaluation(df, target_week_range, weeks_before_target, training_window=120, method='random_forest'); print("\n")

    # print("LightGBM Ranker:") ; run_evaluation(df, target_week_range, weeks_before_target, training_window=180, method='lgbm_ranker'); print("\n")

    # print("XGBoost Ranker:") ; run_evaluation(df, target_week_range, weeks_before_target, training_window=300, method='xgboost_ranker'); print("\n")

    # print("CatBoost Ranker:") ; run_evaluation(df, target_week_range, weeks_before_target, training_window=250, method='catboost_ranker'); print("\n")

    # print("RNN:") ; run_evaluation(df, target_week_range, weeks_before_target, training_window=500, method='rnn'); print("\n")

    # print("LSTM:") ; run_evaluation(df, target_week_range, weeks_before_target, training_window=500, method='lstm'); print("\n")