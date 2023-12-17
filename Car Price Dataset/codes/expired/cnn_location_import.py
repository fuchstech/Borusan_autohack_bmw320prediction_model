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
bmw_data['location_transformed'] = bmw_data['location'].apply(transform_location)

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
features = ['replacement_value', 'paint_value', 'milage', 'year', 'crash cost', 'location_transformed']
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

# Sinir Ağı Dizaynı
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regresyon için çıkış katmanı

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

# Modeli eğitme
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

# Modeli test setinde değerlendirme
test_loss = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Kaybı (MSE): {test_loss}')

# Test seti için tahminler üretme
y_pred_nn = model.predict(X_test_scaled).flatten()

# [Ek metriklerin hesaplanması ve çizim kodları aynı kalıyor]
mae = mean_absolute_error(y_test, y_pred_nn)
mse = test_loss  # Mean Squared Error
rmse = mse ** 0.5  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred_nn)  # R-squared
# Değerlendirme metriklerini yazdırma
print(f'Ortalama Mutlak Hata (MAE): {mae}')
print(f'Kök Ortalama Kare Hata (RMSE): {rmse}')
print(f'R-kare: {r2}')