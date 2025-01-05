import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

def calculate_multiclass_metrics(y_true: pd.Series, y_pred: np.ndarray, classes: list = [1, 2, 3, 4, 5]) -> dict: 
    """
    Calculate MSE, accuracy, TP, TN, FP, FN for multi-class classification.

    Parameters:
    y_true (pd.Series): Ground truth (true values)
    y_pred (np.ndarray): Predicted values
    classes (list): List of unique classes

    Returns:
    dict: Dictionary containing metrics for each class
    """
    # Ensure y_true and y_pred are of the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Calculate MSE
    unrounded_mse = mean_squared_error(y_true, y_pred)
    
    y_pred = np.round(y_pred).astype(int)
    mse = mean_squared_error(y_true, y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics_dict = {"MSE": mse, "Accuracy": accuracy, "Class Metrics": {}, "Unrounded MSE": unrounded_mse}
    
    for cls in classes:
        # Binary conversion for the current class
        y_true_binary = (np.array(y_true) == cls).astype(int)
        y_pred_binary = (np.array(y_pred) == cls).astype(int)
        
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        metrics_dict["Class Metrics"][cls] = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        }
        
    return metrics_dict

# Example usage:
# y_true = pd.Series([1, 0, 1, 1, 0])
# y_pred = np.array([1, 0, 0, 1, 1])
# metrics = calculate_metrics(y_true, y_pred)
# print(metrics)
