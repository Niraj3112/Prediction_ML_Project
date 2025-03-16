import os
import sys
import joblib
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "final_model.pkl")

    def train_model(self, train_arr, test_arr, preprocessor_path, target_scaler_path):
        try:
            logging.info("Loading transformed datasets")

            X_train, y_train_scaled = train_arr
            X_test, y_test_scaled = test_arr

            logging.info("Successfully loaded transformed datasets.")

            logging.info("Initializing KNN model and hyperparameter tuning.")

            knn_params = {
                "n_neighbors": [1, 3],
                "weights": ["distance"],
                "metric": ["euclidean"],
                "p": [1, 2]  # Minkowski parameter (1 = Manhattan, 2 = Euclidean)
            }
            
            knn = KNeighborsRegressor()
            knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='r2', n_jobs=-1)
            knn_grid.fit(X_train, y_train_scaled)

            best_knn = knn_grid.best_estimator_
            logging.info(f"Best KNN Parameters: {knn_grid.best_params_}")

            # Evaluate the model
            train_metrics, test_metrics = evaluate_model(best_knn, X_train, y_train_scaled, X_test, y_test_scaled)
            
            # Print evaluation results
            print("Training Metrics:", train_metrics)
            print("Testing Metrics:", test_metrics)

            # Save the trained model
            joblib.dump(best_knn, self.model_path)
            logging.info("Model training and saving completed.")

            return best_knn, self.model_path

        except Exception as e:
            raise CustomException(e, sys)
