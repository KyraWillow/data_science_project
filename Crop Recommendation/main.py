# --- 1. Persiapan Awal ---
# Mengimpor pustaka-pustaka dasar untuk data, model, dan evaluasi.
# Pustaka utama: pandas (data), scikit-learn (model & evaluasi).
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

"""
0: apple
1: banana
2: blackgram
3: chickpea
4: coconut
5: coffee
6: cotton
7: grapes
8: jute
9: kidneybeans
10: lentil
11: maize
12: mango
13: mothbeans
14: mungbean
15: muskmelon
16: orange
17: papaya
18: pigeonpeas
19: pomegranate
20: rice
21: watermelon
"""

# Memuat dataset yang label tanamannya sudah diubah menjadi angka.
df = pd.read_csv("./data/crop_recommendation_encoded.csv")

# --- 2. Pelatihan dan Evaluasi Model ---
# Memisahkan data menjadi fitur (X) dan target/label (y).
X = df.drop(columns=["label"], axis=1)
y = df["label"]

# Membagi data menjadi set pelatihan (80%) dan set pengujian (20%).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Membuat dan melatih model Regresi Logistik untuk klasifikasi multikelas.
model = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
)
model.fit(X_train, y_train)

# Menguji performa model pada data pengujian asli dan mencetak laporannya.
# Laporan ini menunjukkan seberapa baik model memprediksi tanaman pada data yang belum pernah dilihatnya.
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print("Laporan Klasifikasi pada Data Uji Asli (X_test):")
print(report)

# --- 3. Pengujian Model dengan Data Baru (Skenario Buatan) ---
# Menyiapkan beberapa sampel data baru dengan nilai fitur yang ditentukan manual.
# Ini untuk melihat bagaimana model bereaksi terhadap input spesifik.
new_test_data_values = {
    "N": [50, 50, 90, 20, 120],
    "P": [60, 60, 45, 15, 70],
    "K": [70, 70, 45, 10, 150],
    "temperature": [28.5, 28.5, 22.0, 35.0, 25.0],
    "humidity": [75.2, 75.2, 80.0, 60.0, 90.0],
    "ph": [6.2, 6.2, 6.8, 5.5, 7.1],
    "rainfall": [110.5, 110.5, 150.0, 60.0, 200.0],
}
df_new_test = pd.DataFrame(new_test_data_values)
# Memastikan kolom data baru sesuai dengan yang digunakan saat pelatihan.
df_new_test = df_new_test[X_train.columns]

print("\nData Uji Baru yang Akan Digunakan:")
print(df_new_test)

# Menggunakan model yang sudah dilatih untuk memprediksi label tanaman pada data baru tersebut.
new_predictions = model.predict(df_new_test)

# Menampilkan hasil prediksi untuk setiap sampel data baru.
# Outputnya adalah label angka yang bisa diinterpretasikan menggunakan referensi manual di atas.
print("\nHasil Prediksi untuk Data Uji Baru:")
for i, prediction in enumerate(new_predictions):
    print(
        f"Data ke-{i+1}: Input = {df_new_test.iloc[i].to_dict()}, Prediksi Label Tanaman = {prediction}"
    )
