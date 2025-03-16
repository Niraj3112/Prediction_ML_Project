import logging
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException  # Import your custom exception

def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        logging.info("Evaluating model performance...")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = {
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "MSE": mean_squared_error(y_train, y_train_pred),
            "R2": r2_score(y_train, y_train_pred),
        }

        test_metrics = {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "R2": r2_score(y_test, y_test_pred),
        }

        logging.info(f"Train Metrics: {train_metrics}")
        logging.info(f"Test Metrics: {test_metrics}")

        return train_metrics, test_metrics

    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys) 
