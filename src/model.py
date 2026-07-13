# This file defines and builds the LSTM neural network architecture, used as the core forecasting model that learns temporal patterns in airline passenger data.

 
"""
====================================================
Module : model.py
Project: Airline Passenger Forecasting
Purpose: Build Deep Learning Models
====================================================
"""
 
from tensorflow.keras.models import Sequential
 
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    SimpleRNN,
    GRU,
    Dropout
)
 
 
class ModelBuilder:
 
    """
    Build different RNN models.
    """
 
    def __init__(self,
                 model_type="lstm",
                 input_shape=(12,1)):
        """
        Parameters
        ----------
        model_type : str
            rnn / lstm / gru
 
        input_shape : tuple
            (timesteps, features)
        """
 
        self.model_type = model_type.lower()
        self.input_shape = input_shape
 
    def build_model(self):
 
        model = Sequential()
 
        # -------------------------
        # First Recurrent Layer
        # -------------------------
 
        if self.model_type == "rnn":
 
            model.add(
                SimpleRNN(
                    units=64,
                    input_shape=self.input_shape
                )
            )
 
        elif self.model_type == "gru":
 
            model.add(
                GRU(
                    units=64,
                    input_shape=self.input_shape
                )
            )
 
        else:
 
            model.add(
                LSTM(
                    units=64,
                    input_shape=self.input_shape
                )
            )
 
        # -------------------------
        # Dropout Layer
        # -------------------------
 
        model.add(
            Dropout(0.20)
        )
 
        # -------------------------
        # Hidden Dense Layer
        # -------------------------
 
        model.add(
            Dense(
                units=32,
                activation="relu"
            )
        )
 
        # -------------------------
        # Output Layer
        # -------------------------
 
        model.add(
            Dense(
                units=1
            )
        )
 
        # -------------------------
        # Compile Model
        # -------------------------
 
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )
 
        print("\nModel Built Successfully.\n")
 
        model.summary()
 
        return model
if __name__ == "__main__":
 
    builder = ModelBuilder(
        model_type="lstm",
        input_shape=(12,1)
    )
 
    model = builder.build_model()
 