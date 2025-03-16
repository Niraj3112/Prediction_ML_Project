# Prediction ML Project

## Project Overview
This project is a Machine Learning (ML) pipeline designed to predict **mycotoxin levels (DON concentration)** in corn samples using **hyperspectral imaging data**. The system includes data preprocessing, regression model training, evaluation, and deployment through an interactive **Streamlit** web application.

## Problem Statement
The goal is to build a predictive model using hyperspectral data to estimate **DON concentration** in corn. Each sample has spectral reflectance values at different wavelengths, and the model learns patterns to make accurate predictions.

## Project Structure
```
PREDICTION_ML_PROJECT/
│── notebooks/              # Jupyter notebooks for data analysis and experimentation
│   ├── best_model.pkl      # Best performing model
│   ├── data.csv            # Dataset
│   ├── EDA.ipynb           # Exploratory Data Analysis (EDA)
│   ├── finalmodel.pkl      # Final trained model
│   ├── pca.pkl             # PCA transformation (if used)
│   ├── target_scaler.pkl   # Target variable scaler
│
│── src/                    # Source code for the project
│   ├── components/         # Contains modular ML components
│   │   ├── __init__.py     # Init file for components module
│   │   ├── data_ingestion.py # Loads and processes raw data (Triggers full pipeline)
│   │   ├── data_transformation.py # Data preprocessing and feature engineering
│   │   ├── model_trainer.py # Model training and evaluation
│   ├── pipeline/           # End-to-end ML pipeline scripts
│   │   ├── __init__.py     # Init file for pipeline module
│   │   ├── predict_pipeline.py # Prediction pipeline script
│   ├── __init__.py         # Init file for src
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging utilities
│   ├── utils.py            # Utility functions
│
│── application.py          # Streamlit web application for model inference
│
│── venv/                   # Virtual environment (if applicable)
│── .gitignore              # Ignore unnecessary files in Git
│── README.md               # Project documentation
│── requirements.txt        # Dependencies and package requirements
│── setup.py                # Project setup script
```

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo.git
   cd PREDICTION_ML_PROJECT
   ```
2. **Create a Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Install the Project as a Package:**
   ```bash
   pip install -e .
   ```
   The `-e .` flag installs the package in **editable mode**, allowing immediate application of code changes.

4. **Verify Installation:**
   ```bash
   pip list | grep mlproject
   ```
   Expected output:
   ```
   mlproject        0.0.1
   ```

5. **Run the Full ML Pipeline (Data Ingestion, Transformation, and Training):**
   ```bash
   python src/components/data_ingestion.py
   ```
   This step automatically triggers **data transformation** and **model training**.

6. **Run the Prediction Pipeline:**
   ```bash
   python src/pipeline/predict_pipeline.py
   ```

7. **Launch the Streamlit Application:**
   ```bash
   streamlit run application.py
   ```

## How to Use

1. **Prepare Your Input Data:**  
   - Ensure your CSV file follows the expected format with spectral reflectance values as features and no missing data.
   - The first row should contain column names.

2. **Upload CSV in the Streamlit App:**  
   - Run the application:  
     ```bash
     streamlit run application.py
     ```
   - Use the provided file uploader in the app to upload your CSV.

3. **Get Predictions:**  
   - The model will process the file and output predicted **DON concentration** values.

## Features
- **Data Preprocessing:** Handles missing values, feature scaling, and transformation.
- **Feature Engineering:** Uses PCA and spectral filtering techniques.
- **Model Training:** Trains regression models 
- **Hyperparameter Optimization:** Uses GridSearch 
- **Evaluation Metrics:** RMSE, MAE, R² Score.
- **Interactive Web App:** Uses Streamlit for real-time DON concentration predictions.
- **Logging and Exception Handling:** Ensures scalability and debugging support.

## Technologies Used
- Python
- Scikit-learn
- CatBoost
- Pandas & NumPy
- Matplotlib & Seaborn
- **Streamlit** (for the web interface)
