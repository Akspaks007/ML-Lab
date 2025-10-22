import numpy as np
from collections import Counter
from typing import Any, List, Union

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    
    def __init__(self, k: int = 3):
        
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        
        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, x: np.ndarray) -> Any:
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        
        predictions: List[Any] = [self._predict(x) for x in X_test]
        
        return np.array(predictions)
        
if __name__ == '__main__':
    X_train_dummy = np.array([[1, 1], [1, 2], [2, 2], [5, 5]])
    y_train_dummy = np.array(['A', 'A', 'B', 'B'])
    X_test_dummy = np.array([[1.5, 1.5], [6, 6]])
    
    # Test k=1
    knn_1 = KNNClassifier(k=1)
    knn_1.fit(X_train_dummy, y_train_dummy)
    preds_1 = knn_1.predict(X_test_dummy)
    print(f"k=1 Predictions: {preds_1} (Expected: ['A', 'B'])")
    
    # Test k=3
    knn_3 = KNNClassifier(k=3)
    knn_3.fit(X_train_dummy, y_train_dummy)
    preds_3 = knn_3.predict(X_test_dummy)
    print(f"k=3 Predictions: {preds_3} (Expected: ['A', 'B'])")