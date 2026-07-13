# This file evaluates the trained LSTM model's performance using metrics such as RMSE and MAE, used to measure how accurately the model forecasts airline passenger counts.

# RMSE, MAE, MSE
 
"""
====================================================
Module : evaluate.py
Project: Airline Passenger Forecasting
Purpose: Evaluate LSTM Model Performance
====================================================
"""
 
import numpy as np
 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
 
from src.predict import Predictor
from src.utils import resolve_path
 
 
class Evaluator:
 
    """
    Evaluate model performance.
    """
 
    def __init__(self):
        pass
 
    def evaluate(self):
 
        # -----------------------------
        # Generate Predictions
        # -----------------------------
 
        predictor = Predictor()
 
        actual, predicted = predictor.predict()
 
        # -----------------------------
        # Calculate Metrics
        # -----------------------------
 
        mae = mean_absolute_error(
            actual,
            predicted
        )
 
        mse = mean_squared_error(
            actual,
            predicted
        )
 
        rmse = np.sqrt(mse)
 
        print("\nModel Evaluation")
 
        print(f"\nMAE  : {mae:.4f}")
 
        print(f"MSE  : {mse:.4f}")
 
        print(f"RMSE : {rmse:.4f}")
 
        # -----------------------------
        # Save Metrics
        # -----------------------------
 
        metrics_file_path = resolve_path("outputs/metrics.txt")
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
 
        with open(
            metrics_file_path,
            "w"
        ) as file:
 
            file.write("Model Evaluation\n")
 
            file.write("=====================\n")
 
            file.write(f"MAE  : {mae:.4f}\n")
 
            file.write(f"MSE  : {mse:.4f}\n")
 
            file.write(f"RMSE : {rmse:.4f}\n")
 
        print("\nMetrics Saved Successfully.")
 
        return mae, mse, rmse
   
if __name__ == "__main__":
 
    evaluator = Evaluator()
 
    mae, mse, rmse = evaluator.evaluate()
 
 
 