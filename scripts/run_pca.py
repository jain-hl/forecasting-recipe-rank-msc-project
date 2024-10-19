import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_preprocessing import load_data, preprocess_data

def apply_pca(df, n_components=0.95, target_column='recipe_rank', id_column='item_id', week_column='menu_week'):
    # Prepare the data for PCA by removing target and ID columns
    df_pca_pre = df.drop(columns=[target_column, id_column])

    # Standardize the data
    scaler = StandardScaler()
    df_pca_pre_scaled = scaler.fit_transform(df_pca_pre)

    # Perform PCA to retain the specified variance
    pca = PCA(n_components=n_components)
    df_pca_transformed = pca.fit_transform(df_pca_pre_scaled)

    # Convert the PCA result back to a DataFrame
    pca_columns = [f'PC{i+1}' for i in range(df_pca_transformed.shape[1])]
    df_pca = pd.DataFrame(df_pca_transformed, columns=pca_columns, index=df_pca_pre.index)

    # Combine the results back with the untouched columns
    df_final = pd.concat([df_pca, df[[week_column, id_column, target_column]]], axis=1)

    return df_final, pca

def plot_explained_variance(pca):
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/extended_training_df_619.json'
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Apply PCA
    df_pca, pca_model = apply_pca(df)
    
    # Plot the explained variance
    plot_explained_variance(pca_model)