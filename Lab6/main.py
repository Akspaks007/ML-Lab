import numpy as np
import matplotlib.pyplot as plt
import data
import utils
from knn_classifier import KNNClassifier
from typing import List, Tuple

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

def run_evaluation_pipeline(X: np.ndarray, y: np.ndarray, title: str, k_value: int = 3, random_state: int = 42) -> float:
    
    print(f"\nRunning Evaluation on {title} Dataset (k={k_value})")
    
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.2, random_state=random_state) # [cite: 92]

    knn = KNNClassifier(k=k_value)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

def hyperparameter_tuning_and_analysis(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    
    k_values: List[int] = [1, 3, 5, 7, 9, 11, 15]
    accuracies: List[float] = []
    
    print("\nHyperparameter Tuning: Accuracy vs. k-value")
    
    for k in k_values:
        acc = run_evaluation_pipeline(X, y, "Iris (Tuning)", k_value=k)
        accuracies.append(acc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
    
    plt.title('KNN Classifier Accuracy vs. k-value (Iris Dataset)')
    plt.xlabel('k-value (Number of Neighbors)')
    plt.ylabel('Classification Accuracy')
    plt.xticks(k_values) 
    plt.grid(True)
    plt.show()
    
    best_k_index = np.argmax(accuracies)
    best_k = k_values[best_k_index]
    best_accuracy = accuracies[best_k_index]
    
    print(f"\nAnalysis:")
    print(f"Best k-value identified: k={best_k} with an accuracy of {best_accuracy:.4f}")

    return best_k, best_accuracy

if __name__ == '__main__':
    X_iris, y_iris, _, _ = data.load_iris_data()

    print("Initial Evaluation: Iris Dataset (k=3)")
    accuracy_k3 = run_evaluation_pipeline(X_iris, y_iris, "Iris (Initial)", k_value=3)
    
    best_k, _ = hyperparameter_tuning_and_analysis(X_iris, y_iris)
    
    X_wine, y_wine = data.load_wine_data() 
    
    accuracy_wine = run_evaluation_pipeline(X_wine, y_wine, "Wine (Generalization)", k_value=best_k)
    
    print(f"\nFinal Wine Dataset Accuracy (with best k={best_k}): {accuracy_wine:.4f}")