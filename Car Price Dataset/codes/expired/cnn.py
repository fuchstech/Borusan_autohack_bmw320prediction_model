import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
file_path = r'C:\Users\dest4\Desktop\autohackmatiricie\Car Price Dataset\data\BMW.xlsx'  # Replace with your file path
bmw_data = pd.read_excel(file_path)

# Define mappings for replacements and paints
replacement_values = {
    'Engine bonnet': 2,
    'Rear Bumper': 1.5,
    'Front Bumper': 1.5,
    'Left Front Door': 1,
    'Left Front Fender': 0.7,
    'Right Front Fender': 0.7,
    'Right Front Door': 1,
    'Tailgate': 1.3
}

paint_values = {
    'Right Front Fender': 0.5,
    'Right Front Door': 1,
    'Right Rear Door': 1,
    'Right Rear Fender': 0.5,
    'Left Front Fender': 0.5,
    'Left Front Door': 1,
    'Left Rear Door': 1,
    'Left Rear Fender': 0.5,
    'Tailgate': 1,
    'Roof': 3
}

# Function to calculate the total value for replacements and paints
def calculate_total_value(column, value_dict):
    total_value = 0
    if pd.notna(column):
        items = column.split('\n')
        for item in items:
            item = item.strip().capitalize()  # Capitalize each item
            total_value += value_dict.get(item, 0)  # Add the value, default to 0 if not found
    return total_value

# Applying the transformations to the dataset
bmw_data['replacement_value'] = bmw_data['replacements'].apply(lambda x: calculate_total_value(x, replacement_values))
bmw_data['paint_value'] = bmw_data['paints'].apply(lambda x: calculate_total_value(x, paint_values))

# Selecting relevant features and the target variable
features = ['replacement_value', 'paint_value', 'milage', 'year', 'crash cost']
target = 'price'

# Dropping rows with missing target values and handling any missing values in features
bmw_data_clean = bmw_data.dropna(subset=[target])
X = bmw_data_clean[features].fillna(bmw_data_clean[features].median())
y = bmw_data_clean[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Design
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
# Training the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1, callbacks = [early_stopping])

# Evaluate the model on the test set
test_loss = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Loss: {test_loss}')


#**********************************************************************************************************************
#Plot predictions vs actual
import matplotlib.pyplot as plt
import numpy as np
y_pred_nn = model.predict(X_test_scaled).flatten()

# Selecting a subset of data for plotting for clarity
sample_size = 20  # Adjust as needed
indices = np.arange(sample_size)
actual_sample = y_test.iloc[:sample_size].to_numpy()
predicted_sample = y_pred_nn[:sample_size]

# Plotting the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(indices, actual_sample, label='Actual Prices', marker='o', color='b')
plt.plot(indices, predicted_sample, label='Predicted Prices', marker='x', color='r')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices Comparison')
plt.legend()
plt.show()

mae = mean_absolute_error(y_test, y_pred_nn)
mse = test_loss  # Mean Squared Error
rmse = mse ** 0.5  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred_nn)  # R-squared

print(f'Test Loss (MSE): {test_loss}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')