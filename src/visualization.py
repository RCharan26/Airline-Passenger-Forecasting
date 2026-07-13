# This file handles all visualization tasks such as plotting raw data trends and model predictions, used to provide graphical insights into the dataset and forecast accuracy.

# Graph plotting
 
"""
====================================================
Module : visualization.py
Project: Airline Passenger Forecasting
Purpose: Visualize Model Performance
====================================================
"""
 
import matplotlib.pyplot as plt
 
 
class Visualizer:
 
    """
    Visualize the model performance.
    """
 
    def __init__(self):
        pass
 
    def plot_training_loss(self, history):
        """
        Plot Training Loss and Validation Loss.
        """
 
        plt.figure(figsize=(10,5))
 
        plt.plot(
            history.history["loss"],
            label="Training Loss"
        )
 
        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )
 
        plt.title("Training Loss vs Validation Loss")
 
        plt.xlabel("Epoch")
 
        plt.ylabel("Loss")
 
        plt.legend()
 
        plt.grid(True)
 
        plt.savefig("outputs/loss_curve.png")
 
        plt.show()
 
        print("Training Loss Graph Saved Successfully.")
 
    def plot_predictions(self, actual, predicted):
        """
        Plot Actual vs Predicted Passenger Counts.
        """
 
        plt.figure(figsize=(12,6))
 
        plt.plot(
            actual,
            label="Actual",
            linewidth=2
        )
 
        plt.plot(
            predicted,
            label="Predicted",
            linewidth=2
        )
 
        plt.title("Actual vs Predicted Passenger Count")
 
        plt.xlabel("Time")
 
        plt.ylabel("Passengers")
 
        plt.legend()
 
        plt.grid(True)
 
        plt.savefig("outputs/prediction.png")
 
        plt.show()
 
        print("Prediction Graph Saved Successfully.")
 
    def plot_future_forecast(self, future_values):
        """
        Plot Future Forecast.
        """
 
        plt.figure(figsize=(12,6))
 
        plt.plot(
            future_values,
            marker="o",
            linewidth=2
        )
 
        plt.title("Future Passenger Forecast")
 
        plt.xlabel("Future Months")
 
        plt.ylabel("Passengers")
 
        plt.grid(True)
 
        plt.savefig("outputs/forecast.png")
 
        plt.show()
 
        print("Forecast Graph Saved Successfully.")
 
if __name__ == "__main__":
 
    import numpy as np
 
    class DummyHistory:
 
        history = {
            "loss": [0.30,0.22,0.16,0.11,0.08],
            "val_loss":[0.35,0.26,0.20,0.15,0.10]
        }
 
    history = DummyHistory()
 
    actual = np.array([300,320,350,380,420,450])
 
    predicted = np.array([295,325,345,385,418,455])
 
    future = np.array([470,485,500,520,540,560])
 
    visualizer = Visualizer()
 
    visualizer.plot_training_loss(history)
 
    visualizer.plot_predictions(actual,predicted)
 
    visualizer.plot_future_forecast(future)
 