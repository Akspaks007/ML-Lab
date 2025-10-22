import matplotlib.pyplot as plt
import numpy as np
import data
import itertools
from typing import List, Any

def plot_iris_eda():
    
    X, y, feature_names, target_name = data.load_iris_data() 
    
    species = np.unique(y)
    n_features = X.shape[1]
    
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_plots = len(feature_pairs)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (f1_idx, f2_idx) in enumerate(feature_pairs):
        ax = axes[i]
        
        for s in species:
            X_species = X[y == s]
            
            ax.scatter(
                X_species[:, f1_idx],
                X_species[:, f2_idx],
                label=s,
                alpha=0.7,
                edgecolors='w',
                s=50
            )
            
        f1_name = feature_names[f1_idx]
        f2_name = feature_names[f2_idx]
        
        ax.set_xlabel(f1_name)
        ax.set_ylabel(f2_name)
        ax.set_title(f'{f1_name} vs {f2_name}')
        ax.legend(title=target_name)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_iris_eda()