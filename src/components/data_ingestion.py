import os
import sys
import pandas as pd
import joblib
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("Prediction_Ml_Project\\notebooks\\data.csv")
            logging.info(f"Dataset loaded with shape: {df.shape}")

            if 'hsi_id' in df.columns:
                df = df.drop(columns=['hsi_id'])
                logging.info("Successfully dropped 'hsi_id' column.")
            else:
                logging.info("'hsi_id' column not found in the dataset.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the processed raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Ensure 'hs_id' is also dropped from train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Data ingestion completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_path, target_scaler_path = transformation_obj.initiate_data_transformation(train_data, test_data)

# Model Training
    trainer = ModelTrainer()
    model, model_path = trainer.train_model(train_arr, test_arr, preprocessor_path, target_scaler_path)

    print("Model saved at:", model_path)
    print("Target Scaler saved at:", target_scaler_path)




