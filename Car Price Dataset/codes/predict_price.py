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
import pickle 

loaded_model = pickle.load(open(r"Car Price Dataset\results\ml_model\finalized_model.sav", 'rb'))

file_path = r'Car Price Dataset\data\BMW.xlsx'
bmw_data = pd.read_excel(file_path)
bmw_data_clean = bmw_data.drop(columns=['link', 'location', 'replacements', 'paints', 'extra'])
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

def get_user_input():
    # Kullanıcıdan verileri al
    milage = float(input("Aracın kilometre bilgisini girin: "))
    year = int(input("Aracın üretim yılını girin: "))
    crash_cost = float(input("Aracın geçirdiği kaza maliyetini girin: "))
    replacements = input("Değiştirilen parçaları virgülle ayırarak girin (örn: Motor kaputu, Arka tampon): ")
    paints = input("Boyanan parçaları virgülle ayırarak girin (örn: Sağ ön çamurluk, Sol arka kapı): ")
    location = input("Aracın bulunduğu şehri girin: ")
    def transform_location(location):
        if location in group_0_cities:
            return 0
        elif location in group_1_cities:
            return 1
        else:
            return None  # Eğer listede yoksa None değerini döndürür (veya başka bir değer atayabilirsiniz)

    def calculate_total_value(column, value_dict):
        total_value = 0
        if pd.notna(column):
            items = column.split('\n')
            for item in items:
                item = item.strip().capitalize()  # Capitalize each item
                total_value += value_dict.get(item, 0)  # Add the value, default to 0 if not found
        return total_value
    # Verileri işle
    replacement_value = calculate_total_value(replacements, replacement_values)
    paint_value = calculate_total_value(paints, paint_values)
    location_transformed = transform_location(location)

    # İşlenmiş verileri döndür
    return [replacement_value, paint_value, milage, year, crash_cost]


# Kullanıcıdan veri al
user_input = get_user_input()
loaded_scaler = pickle.load(open(r'Car Price Dataset\results\ml_model\scaler.sav', 'rb'))

# Veriyi ölçeklendirme ve model girdisi olarak hazırlama
user_input_scaled = loaded_scaler.transform([user_input])

# Model tahminini yapma (örnek olarak SVM modeli kullanıldı)
predicted_price = loaded_model.predict(user_input_scaled)
print(f"Tahmini fiyat: {predicted_price[0]}")
