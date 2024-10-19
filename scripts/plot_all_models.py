import json
import matplotlib.pyplot as plt
import numpy as np

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__":
    # Combining all data from json file
    with open('data/long_term_results.json', 'r') as f:
        long_term_results = json.load(f)

    # Calculate the mean of MAEs for each recipe (method)
    mean_maes = {
        recipe: np.mean([item['average_mean_absolute_error'] for item in values])
        for recipe, values in long_term_results.items()
    }

    # Sort the methods by the mean of MAEs in descending order
    sorted_methods = sorted(mean_maes.items(), key=lambda x: x[1], reverse=True)

    # Initialize the plot
    plt.figure(figsize=(15, 6))
    colormap = plt.colormaps.get_cmap('tab20')
    colors = colormap(np.linspace(0, 1, len(sorted_methods)))
    for i, (recipe, _) in enumerate(sorted_methods):
        values = long_term_results[recipe]
        weeks = [item['week_before_target'] for item in values]
        errors = [item['average_mean_absolute_error'] for item in values]
        
        plt.plot(weeks, errors, marker='x', label=f"{recipe} (Mean MAE: {mean_maes[recipe]:.4f})", color=colors[i])

    plt.xlabel('Week Before Target')
    plt.ylabel('Average Mean Absolute Error')
    plt.title('Combined Plot of Average MAEs Across Weeks for All Long Term Methods')
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()