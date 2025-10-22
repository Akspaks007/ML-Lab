import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
from typing import Tuple

X = None
y = None
feature_names = None
target_name = None

def load_iris_data() -> Tuple[np.ndarray, np.ndarray, list, str]:
    
    global X, y, feature_names, target_name
    
    iris = fetch_ucirepo(id=53)

    X_df = iris.data.features 
    y_df = iris.data.targets 

    feature_names = X_df.columns.tolist()
    target_name = y_df.columns[0]
    
    target_col = y_df.columns[0]
    y_df[target_col] = y_df[target_col].str.replace('Iris-', '') 

    X = X_df.values
    y = y_df.values.flatten()

    print("Iris Data Loaded and Preprocessed.")
    return X, y, feature_names, target_name


def load_wine_data() -> Tuple[np.ndarray, np.ndarray]:
    
    wine = fetch_ucirepo(id=109) 

    X_df = wine.data.features 
    y_df = wine.data.targets 

    X_wine = X_df.values
    y_wine = y_df.values.flatten()
    
    print("Wine Data Loaded.")
    return X_wine, y_wine

if __name__ == '__main__':
    X_iris, y_iris, _, _ = load_iris_data()
    print(f"\nIris Features (X) shape: {X_iris.shape}")
    print(f"Iris Labels (y) shape: {y_iris.shape}")
    print(f"First 5 Iris features:\n{X_iris[:5]}")
    print(f"First 5 Iris labels: {y_iris[:5]}")

    X_wine, y_wine = load_wine_data()
    print(f"\nWine Features (X) shape: {X_wine.shape}")
    print(f"Wine Labels (y) shape: {y_wine.shape}")