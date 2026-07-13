# This file converts the preprocessed time series data into fixed-length input sequences, used to structure the data into the sliding-window format required by the LSTM model.

# Create time-series sequences
"""
====================================================
Module : sequence_generator.py
Project: Airline Passenger Forecasting
Purpose: Generate sequences for RNN/LSTM
====================================================
"""
 
import numpy as np
 
 
class SequenceGenerator:
    """
    Convert a time series into input-output sequences.
    """
 
    def __init__(self, sequence_length=12):
        """
        Parameters
        ----------
        sequence_length : int
            Number of previous time steps used
            to predict the next value.
        """
 
        self.sequence_length = sequence_length
 
    def create_sequences(self, scaled_df):
 
        X = []
        y = []
 
        # Convert dataframe to numpy array
        data = scaled_df.values
 
        # Sliding Window
        for i in range(len(data) - self.sequence_length):
 
            X.append(data[i:i+self.sequence_length])
 
            y.append(data[i+self.sequence_length])
 
        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)
 
        print("\nSequence Generation Completed.")
 
        print(f"Input Shape : {X.shape}")
 
        print(f"Output Shape : {y.shape}")
 
        return X, y
 
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
 
    from src.utils import resolve_path

    DATA_PATH = resolve_path("data/airline_passengers.csv")

    # Load dataset
    loader = DataLoader(DATA_PATH)

    df = loader.load_data()
 
    # Scale dataset
    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)
 
    # Generate sequences
    generator = SequenceGenerator(sequence_length=12)
 
    X, y = generator.create_sequences(scaled_df)
 
    print("\nFirst Input Sequence\n")
    print(X[0])
 
    print("\nFirst Target\n")
    print(y[0])
 
# Input Shape : (132, 12, 1)
# Output Shape : (132, 1)
 
# 132   Number of training samples
# 12    Time steps (12 months)
# 1 One feature (Passengers)
 
 