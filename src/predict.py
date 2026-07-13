# This file uses the trained LSTM model to make predictions on test data, used to generate the model's output before evaluation and visualization.

# Predictions
 
"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""
 
import joblib
import numpy as np
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
 
 
class Predictor:
 
    def __init__(self):
 
        self.data_path = "data/airline_passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
    def predict(self):
 
        # ----------------------------
        # Load Dataset
        # ----------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # ----------------------------
        # Scale Dataset
        # ----------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # ----------------------------
        # Generate Sequences
        # ----------------------------
 
        generator = SequenceGenerator(sequence_length=12)
 
        X, y = generator.create_sequences(scaled_df)
 
        # ----------------------------
        # Train Test Split
        # ----------------------------
 
        splitter = TimeSeriesSplit(train_size=0.80)
 
        X_train, X_test, y_train, y_test = splitter.split(X, y)
 
        # ----------------------------
        # Load Model
        # ----------------------------
 
        model = load_model(self.model_path)
 
        print("\nModel Loaded Successfully.")
 
        # ----------------------------
        # Load Scaler
        # ----------------------------
 
        scaler = joblib.load(self.scaler_path)
 
        print("Scaler Loaded Successfully.")
 
        # ----------------------------
        # Predict
        # ----------------------------
 
        predictions = model.predict(X_test)
 
        # ----------------------------
        # Convert back to original scale
        # ----------------------------
 
        predictions = scaler.inverse_transform(predictions)
 
        y_test = scaler.inverse_transform(y_test)
 
        print("\nPrediction Completed.")
 
        return y_test, predictions
if __name__ == "__main__":
 
    predictor = Predictor()
 
    actual, predicted = predictor.predict()
 
    print("\nFirst 10 Predictions\n")
 
    for i in range(10):
 
        print(
            f"Actual : {actual[i][0]:.2f}   "
            f"Predicted : {predicted[i][0]:.2f}"
        )
 