import pandas as pd
import shap

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_preprocessing import load_data, preprocess_data
from src.models import (
    get_linear_regression_model, 
    get_lightgbm_model, 
    get_xgboost_model, 
    get_random_forest_model, 
    get_lgbm_ranker_model, 
    get_xgboost_ranker_model, 
    get_catboost_ranker_model
)

# Run SHAP analysis for each model on training data
def run_evaluation_shap(df, training_window, method='xgboost'):
    """
    Evaluates the SHAP values for a given model on training data.
    
    Parameters:
    - df (pd.DataFrame): The data frame to train the model and perform SHAP analysis on.
    - training_window (int): Size of the training window.
    - method (str): The model to use for SHAP analysis. (Default is 'xgboost')
    
    Returns:
    - pd.DataFrame: SHAP values for the model's features.
    """
    df = df.copy()
    
    method_dict = {
        'linear_regression': get_linear_regression_model,
        'lightgbm': get_lightgbm_model,
        'xgboost': get_xgboost_model,
        'random_forest': get_random_forest_model,
        'lgbm_ranker': get_lgbm_ranker_model,
        'xgboost_ranker': get_xgboost_ranker_model,
        'catboost_ranker': get_catboost_ranker_model,
    }

    rank_percentile_models = ['linear_regression', 'lightgbm', 'xgboost', 'random_forest', 'catboost_ranker']

    # Feature engineering: rank percentile for rank_percentile_models
    if method in rank_percentile_models:
        df['recipe_rank'] = df.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=True)

    # Model function
    model_func = method_dict[method]

    # Training data window
    menu_week_index = max(df['menu_week']) - 1
    train_df = df[(df['menu_week'] >= max(menu_week_index - training_window, 0)) & (df['menu_week'] <= menu_week_index)]

    X_train = train_df.drop(columns=['recipe_rank'])
    y_train = train_df['recipe_rank']

    # Train the model
    trained_model = model_func(X_train, y_train)

    # Use SHAP TreeExplainer for tree-based models, LinearExplainer for linear models, KernelExplainer otherwise
    if method in ['xgboost', 'lightgbm', 'random_forest']:
        explainer = shap.TreeExplainer(trained_model)
    elif method == 'linear_regression':
        explainer = shap.LinearExplainer(trained_model, X_train)
    else:
        explainer = shap.KernelExplainer(trained_model.predict, X_train)
    
    # SHAP values for the training data
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=X_train.columns)

    # Calculate the mean absolute SHAP values and rank features
    avg_shap_values = shap_df.abs().mean().sort_values(ascending=False)

    # Print the top 15 features and their average SHAP values
    print(f"\nTop 15 features based on SHAP values for {method}:")
    print(avg_shap_values.head(15))

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_train)

    # Return SHAP DataFrame for further analysis
    return shap_df

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Run SHAP analysis for different models
    print("Running SHAP analysis for Linear Regression:")
    shap_values_linear_regression = run_evaluation_shap(df, training_window=50, method='linear_regression')

    print("\nRunning SHAP analysis for LightGBM:")
    shap_values_lightgbm = run_evaluation_shap(df, training_window=100, method='lightgbm')

    print("\nRunning SHAP analysis for XGBoost:")
    shap_values_xgboost = run_evaluation_shap(df, training_window=300, method='xgboost')

    print("\nRunning SHAP analysis for Random Forest:")
    shap_values_random_forest = run_evaluation_shap(df, training_window=120, method='random_forest')

    print("\nRunning SHAP analysis for LightGBM Ranker:")
    shap_values_lightgbm_ranker = run_evaluation_shap(df, training_window=180, method='lgbm_ranker')

    print("\nRunning SHAP analysis for XGBoost Ranker:")
    shap_values_xgboost_ranker = run_evaluation_shap(df, training_window=300, method='xgboost_ranker')

    print("\nRunning SHAP analysis for CatBoost Ranker:")
    shap_values_catboost_ranker = run_evaluation_shap(df, training_window=250, method='catboost_ranker')
