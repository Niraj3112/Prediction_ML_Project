import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
        self.target_scaler_file_path = os.path.join("artifacts", "scaled.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info("📥 Reading train and test datasets")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values.reshape(-1, 1)  # Reshaped for scaling
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

            logging.info("Applying StandardScaler on X")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logging.info("Applying PCA on X")
            pca = PCA(n_components=50)  
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            logging.info("Applying StandardScaler on Y (Target Variable)")
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(y_train).ravel()
            y_test_scaled = target_scaler.transform(y_test).ravel()

            logging.info("Saving preprocessors (Scaler + PCA) and target scaler")
            joblib.dump({"scaler": scaler, "pca": pca}, self.data_transformation_config.preprocessor_obj_file_path)
            joblib.dump(target_scaler, self.data_transformation_config.target_scaler_file_path)

            train_arr = (X_train_pca, y_train_scaled)
            test_arr = (X_test_pca, y_test_scaled)

            logging.info("✅ Data transformation completed successfully!")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_scaler_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
