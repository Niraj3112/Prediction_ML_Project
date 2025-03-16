import os
import sys
import joblib
import numpy as np
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.target_scaler_path = os.path.join("artifacts", "scaled.pkl")  # ✅ Load target scaler
        self.model_path = os.path.join("artifacts", "final_model.pkl")

    def predict(self, input_data):
        try:
            logging.info("🔍 Loading preprocessor...")
            preprocessor = joblib.load(self.preprocessor_path)

            if not isinstance(preprocessor, dict) or "scaler" not in preprocessor or "pca" not in preprocessor:
                raise ValueError("❌ Loaded preprocessor is not a valid transformer!")

            scaler_X = preprocessor["scaler"]
            pca = preprocessor["pca"]

            logging.info("🔍 Loading trained model...")
            model = joblib.load(self.model_path)

            if not hasattr(model, "predict"):
                raise ValueError("❌ Loaded model is not a valid predictor!")

            logging.info("⚙️ Applying StandardScaler transformation on input...")
            input_scaled = scaler_X.transform(input_data)

            logging.info("⚙️ Applying PCA transformation on input...")
            input_pca = pca.transform(input_scaled)

            logging.info("🤖 Making predictions...")
            y_scaled_pred = model.predict(input_pca)

            logging.info("🔄 Loading target scaler for inverse transform...")
            if os.path.exists(self.target_scaler_path):
                scaler_y = joblib.load(self.target_scaler_path)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()
            else:
                logging.warning("⚠️ Target scaler not found! Returning raw scaled predictions.")
                y_pred = y_scaled_pred  # Return scaled prediction if inverse transform is unavailable

            logging.info(f"✅ Final Prediction: {y_pred}")
            return y_pred

        except Exception as e:
            raise CustomException(e, sys)

# Main script to test with random input
if __name__ == "__main__":
    import numpy as np
    from src.pipeline.predict_pipeline import PredictPipeline

    # Generate Random Test Data (448 features)
    random_input = np.random.rand(1, 448)  # Simulating a single-row CSV input

    # Initialize Prediction Pipeline
    pipeline = PredictPipeline()

    # Run Prediction
    prediction = pipeline.predict(random_input)

    # Output Prediction
    print(f"🌾 Predicted Crop Yield: {prediction[0]:.2f}")