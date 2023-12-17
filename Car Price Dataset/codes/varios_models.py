import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

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
"""
group_0_cities = ["Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Ardahan", 
                  "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl", 
                  "Bitlis", "Bolu", "Burdur", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Elazığ", 
                  "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", 
                  "Iğdır", "Isparta", "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri", 
                  "Kilis", "Kırıkkale", "Kırklareli", "Kırşehir", "Konya", "Kütahya", "Malatya", "Manisa", 
                  "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Osmaniye", "Rize", 
                  "Samsun", "Siirt", "Sinop", "Sivas", "Şanlıurfa", "Şırnak", "Tekirdağ", "Tokat", "Trabzon", 
                  "Tunceli", "Uşak", "Van", "Yozgat", "Zonguldak"]

group_1_cities = ["Bursa", "Kocaeli", "Sakarya", "Yalova", "Tekirdağ", "Edirne", "Çanakkale", "İzmir"]

# location sütununu dönüştürecek fonksiyon
def transform_location(location):
    if location in group_0_cities:
        return 0
    elif location in group_1_cities:
        return 1
    else:
        return None  # Eğer listede yoksa None değerini döndürür (veya başka bir değer atayabilirsiniz)

# Uygulama
bmw_data['location_transformed'] = bmw_data['location'].apply(transform_location)"""

# Function to calculate the total value for replacements and paints
def calculate_total_value(column, value_dict):
    total_value = 0
    if pd.notna(column):
        items = column.split('\n')
        for item in items:
            item = item.strip().capitalize()  # Capitalize each item
            total_value += value_dict.get(item, 0)  # Add the value, default to 0 if not found
    return total_value

bmw_data['replacement_value'] = bmw_data['replacements'].apply(lambda x: calculate_total_value(x, replacement_values))
bmw_data['paint_value'] = bmw_data['paints'].apply(lambda x: calculate_total_value(x, paint_values))

# 'location_transformed' özelliğini de dahil ediyoruz
features = ['replacement_value', 'paint_value', 'milage', 'year', 'crash cost']
target = 'price'

# Verileri temizleme ve özellik değerlerini doldurma
bmw_data_clean = bmw_data.dropna(subset=[target])
X = bmw_data_clean[features].fillna(bmw_data_clean[features].median())
y = bmw_data_clean[target]

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# SVM Modeli
svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM MSE:", mean_squared_error(y_test, y_pred_svm))
print("SVM R2:", r2_score(y_test, y_pred_svm))

# Doğrusal Regresyon Modeli
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# Polinom Regresyon Modeli
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_train_scaled, y_train)
y_pred_poly = poly_model.predict(X_test_scaled)
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression R2:", r2_score(y_test, y_pred_poly))

# Rastgele Orman Modeli
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# K-En Yakın Komşu Modeli
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))

import matplotlib.pyplot as plt
import numpy as np

# Örnek sayısı (grafikte gösterilecek veri sayısı)
sample_size = 20

# Gerçek ve tahmin edilen değerleri almak için indeksler
indices = np.arange(sample_size)

# Test setinden örnekler
actual_sample = y_test.iloc[:sample_size]

# Her model için tahminleri al ve grafik çiz
models = {'SVM': y_pred_svm, 'Linear Regression': y_pred_lr, 
          'Polynomial Regression': y_pred_poly, 'Random Forest': y_pred_rf, 
          'KNN': y_pred_knn}

for model_name, y_pred in models.items():
    plt.figure(figsize=(12, 6))
    plt.plot(indices, actual_sample, label='Actual Prices', marker='o', color='blue')
    plt.plot(indices, y_pred[:sample_size], label=f'Predicted Prices ({model_name})', marker='x', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title(f'Actual vs Predicted Prices Comparison - {model_name}')
    plt.legend()
    plt.show()
