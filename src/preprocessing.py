#  performs exploratory data analysis (EDA) and preprocessing steps such as scaling and cleaning, used to prepare raw airline passenger data for model training.

# Scaling & preprocessing
 
"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""
 
# Import required libraries
import joblib
import pandas as pd
 
from sklearn.preprocessing import MinMaxScaler
from src.utils import resolve_path
 
 
class Preprocessor:
    """
    Preprocess the time series dataset.
    """
 
    def __init__(self):
        """
        Initialize the scaler.
        """
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
 
    def scale_data(self, df):
        """
        Scale the Passengers column.
 
        Parameters
        ----------
        df : pandas.DataFrame
 
        Returns
        -------
        scaled_df : pandas.DataFrame
        """
 
        print("\nOriginal Data")
        print(df.head())
 
        # Scale the Passengers column
        scaled_values = self.scaler.fit_transform(df[["Passengers"]])
 
        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=["Passengers"],
            index=df.index
        )
 
        # print("\nScaled Data")
        # print(scaled_df.head())
 
        # Save the scaler
        scaler_path = resolve_path("models/scaler.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
 
        print(f"\nScaler saved successfully to {scaler_path}.")
 
        return scaled_df
 
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
 
    DATA_PATH = "data/airline_passengers.csv"
 
    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale data
    preprocessor = Preprocessor()
 
    scaled_df = preprocessor.scale_data(df)
 
    print("\nScaled Dataset")
    print(scaled_df.head())