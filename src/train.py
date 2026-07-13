# This file handles the training loop for the LSTM model on the prepared sequence data, used to fit the model's weights so it can accurately forecast passenger numbers.

"""
====================================================
Module : train.py
Project: Airline Passenger Forecasting
Purpose: Train the LSTM Model
====================================================
"""
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
from src.model import ModelBuilder
 
 
class ModelTrainer:
 
    def __init__(self):
 
        self.data_path = "data/airline_passengers.csv"
 
    def train(self):
 
        # ----------------------------
        # Step 1 : Load Dataset
        # ----------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # ----------------------------
        # Step 2 : Preprocess
        # ----------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # ----------------------------
        # Step 3 : Generate Sequences
        # ----------------------------
 
        generator = SequenceGenerator(sequence_length=12)
 
        X, y = generator.create_sequences(scaled_df)
 
        # ----------------------------
        # Step 4 : Train Test Split
        # ----------------------------
 
        splitter = TimeSeriesSplit(train_size=0.80)
 
        X_train, X_test, y_train, y_test = splitter.split(X, y)
 
        # ----------------------------
        # Step 5 : Build Model
        # ----------------------------
 
        builder = ModelBuilder(
            model_type="lstm",
            input_shape=(12,1)
        )
 
        model = builder.build_model()
 
        # ----------------------------
        # Step 6 : Train Model
        # ----------------------------
 
        print("\nTraining Started...\n")
 
        history = model.fit(
 
            X_train,
 
            y_train,
 
            epochs=100,
 
            batch_size=8,
 
            validation_data=(X_test,y_test),
 
            verbose=1
 
        )
 
        print("\nTraining Completed Successfully.")
 
        # ----------------------------
        # Step 7 : Save Model
        # ----------------------------
 
        model.save("models/lstm_model.keras")
 
        print("\nModel Saved Successfully.")
 
        return model, history
   
if __name__ == "__main__":
 
    trainer = ModelTrainer()
 
    model, history = trainer.train()
 
 