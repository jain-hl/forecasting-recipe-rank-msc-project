import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data_short_term
from src.model_evaluation_short_term import run_evaluation_live

if __name__ == "__main__":

    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path, short_term=True)
    df_live = preprocess_data_short_term(df)

    setback = 0
    original_max_week = df['menu_week'].max() + 1 - setback

    # Run the evaluation
    total_week_iterations = 52
    days_before_target = 21
    target_week_range = range(original_max_week - total_week_iterations, original_max_week)

    # print("Linear Regression:"); run_evaluation_live(df_live, target_week_range, days_before_target, method='linear_regression')

    print("LightGBM:"); run_evaluation_live(df_live, target_week_range, days_before_target, method='lightgbm')

    # print("XGBoost:"); run_evaluation_live(df_live, target_week_range, days_before_target, method='xgboost')

    print("Random Forest:"); run_evaluation_live(df_live, target_week_range, days_before_target, method='random_forest')

    print("LightGBM Ranker:") ; run_evaluation_live(df_live, target_week_range, days_before_target, method='lgbm_ranker')

    print("XGBoost Ranker:") ; run_evaluation_live(df_live, target_week_range, days_before_target, method='xgboost_ranker')

    print("CatBoost Ranker:") ; run_evaluation_live(df_live, target_week_range, days_before_target, method='catboost_ranker')