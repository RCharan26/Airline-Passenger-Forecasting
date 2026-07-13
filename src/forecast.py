# This file generates future airline passenger forecasts using the trained model and plots the predicted results, used to produce the final output shown to the user in the app.

# Future forecasting
 
"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""
 
import numpy as np
import joblib
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.utils import resolve_path
 
 
class Forecaster:
 
    def __init__(self):
 
        self.data_path = resolve_path("data/airline_passengers.csv")
 
        self.model_path = resolve_path("models/lstm_model.keras")
 
        self.scaler_path = resolve_path("models/scaler.pkl")
 
        self.sequence_length = 12

 
    def forecast(self, future_months=12):
 
        # --------------------------
        # Load Dataset
        # --------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # --------------------------
        # Scale Dataset
        # --------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # --------------------------
        # Load Model
        # --------------------------
 
        model = load_model(self.model_path)
 
        # --------------------------
        # Load Scaler
        # --------------------------
 
        scaler = joblib.load(self.scaler_path)
 
        # --------------------------
        # Last 12 Months
        # --------------------------
 
        last_sequence = scaled_df.values[-self.sequence_length:]
 
        future_predictions = []
 
        # --------------------------
        # Forecast Loop
        # --------------------------
 
        for _ in range(future_months):
 
            input_data = last_sequence.reshape(
                1,
                self.sequence_length,
                1
            )
 
            prediction = model.predict(
                input_data,
                verbose=0
            )
 
            future_predictions.append(prediction[0,0])
 
            last_sequence = np.vstack(
                (
                    last_sequence[1:],
                    prediction
                )
            )
 
        # --------------------------
        # Convert Back
        # --------------------------
 
        future_predictions = np.array(
            future_predictions
        ).reshape(-1,1)
 
        future_predictions = scaler.inverse_transform(
            future_predictions
        )
 
        print("\nFuture Forecast Completed.")
 
        return future_predictions
if __name__ == "__main__":
 
    forecaster = Forecaster()
 
    future = forecaster.forecast(future_months=12)
 
    print("\nNext 12 Month Forecast\n")
 
    for i, value in enumerate(future, start=1):
 
        print(f"Month {i} : {value[0]:.2f}")
   
 