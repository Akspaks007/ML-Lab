import numpy as np
from typing import Tuple

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if random_state is not None:
        np.random.seed(random_state) 

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    shuffled_indices = np.random.permutation(n_samples)
    
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_dummy = np.arange(10).reshape(10, 1)
    y_dummy = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'a', 'b', 'c', 'a'])

    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42)
    
    print("--- Train-Test Split Test (30% test size, seed 42) ---")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    print(f"X_train (first 5): {X_train.flatten()[:5]}")
    print(f"X_test: {X_test.flatten()}")