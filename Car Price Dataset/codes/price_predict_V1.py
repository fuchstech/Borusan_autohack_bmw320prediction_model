from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR


import pandas as pd

# Load the data from the provided Excel file
file_path = r'C:\Users\dest4\Desktop\autohackmatiricie\Car Price Dataset\data\BMW.xlsx'
bmw_data = pd.read_excel(file_path)

# Display the first few rows of the dataframe to understand its structure
bmw_data.head()

# Dropping non-relevant columns
# Dropping 'link' as it's not relevant for modeling
# 'location', 'replacements', 'paints', 'extra' are textual/categorical, need special handling
bmw_data_clean = bmw_data.drop(columns=['link', 'location', 'replacements', 'paints', 'extra'])

# Handling missing values - Using median for numerical columns
numerical_cols = bmw_data_clean.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    bmw_data_clean[col].fillna(bmw_data_clean[col].median(), inplace=True)

# Defining the features and target variable
X = bmw_data_clean.drop('price', axis=1)
y = bmw_data_clean['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the SVM model
svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)

# Checking the model's score on the training and test sets
train_score = svm_model.score(X_train_scaled, y_train)
test_score = svm_model.score(X_test_scaled, y_test)

print(train_score, test_score)

import matplotlib.pyplot as plt

# Generate predictions
y_pred = svm_model.predict(X_test_scaled)

# Limiting the data points for a clearer plot
sample_size = 20  # number of data points to display
indices = range(sample_size)
print(len(y_pred[:sample_size]), len(y_test[:sample_size]))
# Actual vs Predicted values plot
plt.figure(figsize=(10, 6))
plt.scatter(indices, y_test[:sample_size], color='blue', label='Actual')
plt.scatter(indices, y_pred[:sample_size], color='red', label='Predicted')
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Creating and training the linear regression model with polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predicting on the test set
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluating the model
poly_train_score = poly_model.score(X_train_poly, y_train)
poly_test_score = poly_model.score(X_test_poly, y_test)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = poly_mse ** 0.5

adjusted_sample_size_poly = min(sample_size, len(y_test))

# Selecting a subset of data for plotting
y_test_sample_poly = y_test.iloc[:adjusted_sample_size_poly].to_numpy()
y_pred_sample_poly = y_pred_poly[:adjusted_sample_size_poly]

plt.figure(figsize=(10, 6))
plt.scatter(range(adjusted_sample_size_poly), y_test_sample_poly, color='blue', label='Actual')
plt.scatter(range(adjusted_sample_size_poly), y_pred_sample_poly, color='red', label='Predicted')
plt.title('Comparison of Actual and Predicted Prices (Polynomial Regression)')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


#-------------

import xgboost as xgb

# Creating the XGBoost regressor
# Reconfiguring the XGBoost regressor with reduced complexity
xgb_model_reduced = xgb.XGBRegressor(objective ='reg:squarederror', 
                                     colsample_bytree = 0.3, 
                                     learning_rate = 0.1,
                                     max_depth = 3,  # Reduced max depth
                                     alpha = 10, 
                                     n_estimators = 50)  # Reduced number of estimators

# Training the model with reduced complexity
xgb_model_reduced.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred_xgb_reduced = xgb_model_reduced.predict(X_test_scaled)

# Evaluating the model
xgb_train_score_reduced = xgb_model_reduced.score(X_train_scaled, y_train)
xgb_test_score_reduced = xgb_model_reduced.score(X_test_scaled, y_test)
xgb_mse_reduced = mean_squared_error(y_test, y_pred_xgb_reduced)
xgb_rmse_reduced = xgb_mse_reduced ** 0.5

print(xgb_train_score_reduced, xgb_test_score_reduced, xgb_rmse_reduced)



plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb_reduced, color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()