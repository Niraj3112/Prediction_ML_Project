# Machine Learning Model Report

## **1. Introduction**
This report documents the process of predicting DON (Deoxynivalenol) concentration using spectral reflectance data. Multiple machine learning models were trained, evaluated, and compared. The final model selected was a **K-Nearest Neighbors (KNN) Regressor**, with manually fine-tuned hyperparameters. Principal Component Analysis (PCA) was used for dimensionality reduction, and the final model was trained with **50 principal components**.

---

## **2. Data Preprocessing**
- The dataset was loaded from a CSV file, and the `hsi_id` column was removed.
- Column names were converted to numeric values, and non-numeric columns (like the target variable) were retained separately.
- **No null values or duplicate entries** were found in the dataset.
- Features (X) and the target variable (y) were extracted.
- Data was split into **80% training and 20% testing** using `train_test_split`.
- **Standardization** was applied to both features and target values using `StandardScaler`.
- **PCA was applied with 50 principal components** to reduce dimensionality while retaining important variance.

---

## **3. Model Training & Evaluation**
### **Models Considered:**
1. **Linear Regression**
2. **Lasso Regression**
3. **Ridge Regression**
4. **K-Nearest Neighbors (KNN) Regressor**
5. **Decision Tree Regressor**

### **Performance Metrics Used:**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-Squared Score (R²)**

### **Best Model Selection:**
- After training all models, KNN achieved the best performance.
- PCA was re-applied with **50 components**, leading to optimal feature extraction.
- **Final KNN Model Hyperparameters (Manually Selected):**
  - `n_neighbors = 3`
  - `weights = 'distance'`
  - `metric = 'euclidean'`

---

## **4. Final Model Evaluation**
**Best Model: KNN Regressor**
- **Performance on Test Data:**
  - **MAE:** *Low*
  - **RMSE:** *Low*
  - **R² Score:** *High*
- The predicted DON concentrations closely matched the actual values, as seen in the scatter plot of predicted vs. actual values.

---

## **5. Model Deployment & Future Work**
- The trained KNN model, PCA transformation, and scalers were saved using `joblib` for future inference.
- Future improvements may include:
  - Experimenting with **deep learning techniques** like neural networks.
  - Exploring **feature engineering** for better input representation.
  - Tuning hyperparameters with automated methods like **GridSearchCV**

---

## **6. Conclusion**
- The KNN model with manually optimized hyperparameters and PCA-based dimensionality reduction provided the best results.
- The model is robust and can be used to predict DON concentration effectively based on spectral reflectance data.
- Further improvements could involve deep learning approaches or ensemble methods for enhanced accuracy.

---
**End of Report**

