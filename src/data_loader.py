# This file is responsible for loading the airline passenger dataset from disk, used as the first step in the data pipeline to feed raw data into preprocessing and model training.

 
"""
====================================================
Module : data_loader.py
Project: Airline Passenger Forecasting
Purpose: Load and validate the dataset
====================================================
"""
 
# Import required libraries
import pandas as pd
from pathlib import Path
from src.utils import resolve_path
 
 
class DataLoader:
    """
    A class responsible for loading the time series dataset.
    """
 
    def __init__(self, file_path):
        """
        Constructor
 
        Parameters
        ----------
        file_path : str or Path
            Path of the CSV dataset.
        """
 
        self.file_path = resolve_path(file_path) if isinstance(file_path, str) else Path(file_path)
 
    def load_data(self):
        """
        Load the dataset and return a DataFrame.
 
        Returns
        -------
        pandas.DataFrame
        """
 
        # -----------------------------
        # Step 1 : Check file existence
        # -----------------------------
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at:\n{self.file_path}"
            )
 
        print("Dataset Found Successfully.\n")
 
        # -----------------------------
        # Step 2 : Read CSV
        # -----------------------------
        df = pd.read_csv(self.file_path)
 
        print("Dataset Loaded Successfully.\n")
 
        # -----------------------------
        # Step 3 : Display first rows
        # -----------------------------
        print("First 5 Rows")
        print(df.head())
 
        # -----------------------------
        # Step 4 : Dataset Shape
        # -----------------------------
        print("\nDataset Shape")
        print(df.shape)
 
        # -----------------------------
        # Step 5 : Dataset Information
        # -----------------------------
        print("\nDataset Information")
        print(df.info())
 
        # -----------------------------
        # Step 6 : Missing Values
        # -----------------------------
        print("\nMissing Values")
        print(df.isnull().sum())
 
        # -----------------------------
        # Step 7 : Convert Month column
        # -----------------------------
        df["Month"] = pd.to_datetime(df["Month"])
 
        print("\nData Types After Conversion")
        print(df.dtypes)
 
        # -----------------------------
        # Step 8 : Set Month as Index
        # -----------------------------
        df.set_index("Month", inplace=True)
 
        print("\nMonth column converted into Datetime.")
 
        print("Month column set as Index.\n")
 
        # -----------------------------
        # Step 9 : Return DataFrame
        # -----------------------------
        return df
   
 
# ===========================================
# Test the module
# ===========================================
 
if __name__ == "__main__":
 
    DATA_PATH = "data/airline_passengers.csv"
 
    loader = DataLoader(DATA_PATH)
 
    df = loader.load_data()
 
    print("\nProcessed Dataset")
    print(df.head())