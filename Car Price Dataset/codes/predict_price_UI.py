import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle
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


def predict_price():
    try:
        milage = float(milage_entry.get())
        year = int(year_entry.get())
        crash_cost = float(crash_cost_entry.get())
        replacements = replacements_entry.get()
        paints = paints_entry.get()
        location = location_entry.get()
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
        # Verileri işleme fonksiyonları burada kullanılır
        replacement_value = calculate_total_value(replacements, replacement_values)
        paint_value = calculate_total_value(paints, paint_values)
        location_transformed = transform_location(location)
        loaded_scaler = pickle.load(open(r'Car Price Dataset\results\ml_model\scaler.sav', 'rb'))
        loaded_model = pickle.load(open(r"Car Price Dataset\results\ml_model\finalized_model.sav", 'rb'))

        # Veriyi ölçeklendirme ve model girdisi olarak hazırlama
        user_input_scaled = loaded_scaler.transform([user_input])

        # Model tahminini yapma (örnek olarak SVM modeli kullanıldı)
        predicted_price = loaded_model.predict(user_input_scaled)
        # Tahmin yapma
        user_input = [replacement_value, paint_value, milage, year, crash_cost]
        user_input_scaled = loaded_scaler.transform([user_input])
        predicted_price = loaded_model.predict(user_input_scaled)

        # Tahmini fiyatı göster
        messagebox.showinfo("Tahmini Fiyat", f"Tahmini fiyat: {predicted_price[0]}")
    except ValueError as e:
        messagebox.showerror("Hata", "Lütfen geçerli veriler girin.")

# GUI başlatma
root = tk.Tk()
root.title("Araba Fiyat Tahmini")

# Giriş alanları
tk.Label(root, text="Milage:").grid(row=0, column=0)
milage_entry = tk.Entry(root)
milage_entry.grid(row=0, column=1)

tk.Label(root, text="Year:").grid(row=1, column=0)
year_entry = tk.Entry(root)
year_entry.grid(row=1, column=1)

tk.Label(root, text="Crash Cost:").grid(row=2, column=0)
crash_cost_entry = tk.Entry(root)
crash_cost_entry.grid(row=2, column=1)

tk.Label(root, text="Replacements:").grid(row=3, column=0)
replacements_entry = tk.Entry(root)
replacements_entry.grid(row=3, column=1)

tk.Label(root, text="Paints:").grid(row=4, column=0)
paints_entry = tk.Entry(root)
paints_entry.grid(row=4, column=1)

tk.Label(root, text="Location:").grid(row=5, column=0)
location_entry = tk.Entry(root)
location_entry.grid(row=5, column=1)

# Tahmin butonu
predict_button = tk.Button(root, text="Tahmin Et", command=predict_price)
predict_button.grid(row=6, column=0, columnspan=2)

# GUI başlat
root.mainloop()
