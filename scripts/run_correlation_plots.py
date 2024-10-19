import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data_short_term

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path, short_term=True)
    df_live = preprocess_data_short_term(df)

    # Rank recipe_rank within each menu_week
    df_live_pct = df_live.copy()
    df_live_pct['recipe_rank'] = df_live_pct.groupby('menu_week')['recipe_rank'].rank(method='first', pct=True, ascending=False)

    # Drop uptake_at_lead_day and recipe_rank columns, and compute correlation with the target
    non_uptake_features = df_live_pct.drop(columns=df_live_pct.filter(regex='uptake_at_lead_day|embeddings').columns)
    non_uptake_features = non_uptake_features.drop(columns=['recipe_rank'])
    non_uptake_corr = pd.concat([non_uptake_features, df_live_pct['recipe_rank']], axis=1).corr()

    # Sort features by absolute correlation
    sorted_corr = non_uptake_corr[['recipe_rank']].dropna().reindex(non_uptake_corr[['recipe_rank']].dropna().apply(lambda x: abs(x)).sort_values(by='recipe_rank', ascending=False).index)

    # Plot the most correlated features
    plt.figure(figsize=(10, 6))
    sns.heatmap(sorted_corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlations between non uptake and embedding features and recipe rank percentile')
    plt.tight_layout()
    plt.show()

    # Calculate correlation for uptake_at_lead_day features
    uptake_features = df_live.filter(regex=r'^uptake_at_lead_day')
    uptake_corr = pd.concat([uptake_features, df_live_pct['recipe_rank']], axis=1).corr()

    # Plot correlation for uptake_at_lead_day features
    plt.figure(figsize=(10, 6))
    sns.heatmap(uptake_corr[['recipe_rank']].dropna(), annot=True, fmt='.8f', cmap='coolwarm', vmin=0.90, vmax=1)
    plt.title('Correlation between "uptake_at_lead_day" features and recipe rank percentile')
    plt.tight_layout()
    plt.show()