# House Pricing Regression

## Overview
House Pricing Advanced Regression is a data analysis project aimed at predicting house prices based on various features available in the dataset. This project includes data preprocessing, exploratory data analysis (EDA), data processing, model training, and deployment of regression models for house price prediction.

## Dataset
The dataset consists of two main files:
- `train.csv`: Training data containing 81 features and house sale prices (`SalePrice`).
- `test.csv`: Test data containing 80 features without house sale prices.

## Installation
Make sure you have Python 3.x installed and install the required dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib pickle flask streamlit
```

## Data Preprocessing
### 1. Data Cleaning
- Remove columns with too many missing values.
- Fill in missing values using appropriate methods.

### 2. Handling Outliers
- Use the IQR method to detect and remove outliers.
- Visualize with `sns.boxplot()` for further analysis.

### 3. Normalization and Standardization
- Standardize numerical features using `StandardScaler` from `sklearn.preprocessing`.

### 4. Handling Duplicate Data
- Remove duplicate rows.

### 5. Data Type Conversion
- Use One-Hot Encoding for categorical features.
- Use Label Encoding for features that are better represented as integers.

## Exploratory Data Analysis (EDA)
### 1. Data Distribution
- Analyze distributions using histograms and `describe()`.

### 2. Feature Correlation
- Visualize correlations using heatmaps (`sns.heatmap()`).
- Identify features highly correlated with `SalePrice`.

### 3. Interactive EDA Dashboard
A Streamlit dashboard has been implemented to visualize key insights from the dataset interactively. To run the dashboard, execute the following command:

```bash
streamlit run dashboard.py
```

The dashboard provides various plots and statistics to explore the dataset more effectively.

## Model Training
The models used:
- **Least Angle Regression (LARS)**
- **Linear Regression (LR)**
- **Gradient Boosting Regressor (GBR)**

### 1. Splitting the Dataset
- Split the data into training and testing sets.
- Use `train_test_split()` from `sklearn.model_selection`.

### 2. Training the Models
- Train the models using `Lars`, `LinearRegression`, and `GradientBoostingRegressor`.
- Evaluate the models using metrics such as `R-squared` and `Mean Absolute Error (MAE)`.

### 3. Saving the Models
- The models are saved in `.joblib` and `.pkl` formats for deployment.

```python
import joblib
joblib.dump(lars, 'model/lars_model.joblib')
joblib.dump(LR, 'model/lr_model.joblib')
joblib.dump(GBR, 'model/gbr_model.joblib')

import pickle
with open('model/gbr_model.pkl', 'wb') as file:
    pickle.dump(GBR, file)
```

## Model Deployment
### 1. Loading the Model
```python
import joblib
model = joblib.load('model/gbr_model.joblib')
```

### 2. Running the API with Flask
To test the deployed model, run the Flask API with the following command:

```bash
python deploy.py
```

### 3. Testing the API
Use `cURL` to test the prediction endpoint:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json
```

Ensure that the `data.json` file contains input data in the appropriate JSON format.

## Testing the Deployed Model
- Use the test data (`test.csv`) to evaluate the model's performance.
- Compare predictions with actual prices (if available).
- Adjust model parameters if needed to improve accuracy.

## Conclusion
This project implements regression models to predict house prices with various preprocessing techniques, data exploration, and model evaluation. The developed model can be used in web applications or APIs to provide automatic house price predictions.

Additionally, an interactive Streamlit dashboard has been integrated to facilitate visual exploration of the dataset, making it easier to analyze trends and correlations.