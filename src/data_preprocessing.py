import os
import json
import pandas as pd

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data(file_path, short_term=False):
    # Read JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract fields and records
    fields = [field['name'] for field in data['schema']['fields'] if field['name'] != 'index']
    records = data.get('data', [])
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(records, columns=fields)
    df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
    if short_term:
        df.drop(columns=['contamination_outlier', 'uptake_at_lead_day_-6'], inplace=True)
    return df

def preprocess_data(df):
    # Remove short-term model features (uptake at lead day)
    columns_to_remove = df.filter(like="uptake_at_lead_day").columns
    df.drop(columns=columns_to_remove, inplace=True)
    
    # Remove contamination outlier feature
    if 'contamination_outlier' in df.columns:
        df.drop(columns=['contamination_outlier'], inplace=True)
    
    # Create recipe rank percentile column
    df['recipe_rank'] = df.groupby('menu_week')['recipe_uptake'].rank(method='first', ascending=False) - 1
    df['recipe_rank'] = df['recipe_rank'].astype(int)
    
    # Drop recipe uptake column
    df.drop(columns=['recipe_uptake'], inplace=True)
    
    return df

def preprocess_data_short_term(df):
    df['recipe_rank'] = df.groupby('menu_week')['recipe_uptake'].rank(method='first', ascending=False) - 1
    df['recipe_rank'] = df['recipe_rank'].astype(int)

    columns_to_rank = [f'uptake_at_lead_day_{j}' for j in range(-5, 19)]
    for i, col in enumerate(columns_to_rank):
        df[col] = df.groupby('menu_week')[col].rank(method='first', pct=True, ascending=True)
        if i > 0:
            prev_col = columns_to_rank[i-1]  # Previous column
            df[f'diff_uptake_at_lead_day_{i-6}'] = df[prev_col] - df[col]

    df_live = df[df['menu_week'] >= 488]
    df_live = df_live.drop(columns=['recipe_uptake'])
    return df_live