import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

"""
Program ini bertujuan untuk membuat sebuah model machine learning
deteksi kehidupan orang orang yang mengalamai kecelakaan pada kapal titanic.
"""

# Load dataset
df = pd.read_csv("data\Titanic-Dataset.csv")

# Isi nilai yang hilang dengan median karena lebih tahan/robus dengan outlier
df["Age"] = df["Age"].fillna(df["Age"].median())

# Hapus feature Cabin karena banyak sekali data yang hilang
df.drop(columns=["Cabin"], axis=1, inplace=True)

# Mengisi nilai yang hilang dengan mode (nilai yang sering keluar)
# Mode mengembalikan series, jadi kita ambil element pertama [0]
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Melakukan label encoding untuk feature 'Sex'
# .map() berfungsi untuk mengubah nilai setiap series dengan nilai baru
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)

# Melakukan one hot encoding pada feature 'Embarked'
# One hot encoding akan membbuatkan feature baru yang unik
# prefix berfungsi untuk penamaan awal colom baru berdasarkan isi dari prefix
df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")

# Menghapus feature 'Name' dan 'Ticket'
# Dihapus karena tidak memiliki sinyal prediksi yang kuat
df.drop(["Name", "Ticket"], axis=1, inplace=True)

# Membagi antara fitur dan target
X = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

# Melakukan split data dan trinning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inisiasi models
random_forest = RandomForestClassifier(
    random_state=42
)  # Menentukan tree yang dibangun di dalam forest, contohnya ktia menggunakan 100 desision (umumnya)

# # Hayperparameter Tunning
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "criterion": ["gini", "entropy"],
#     'bootstrap' : [True, False]
# }

# # Inisialisasi GridSearchCV
# grid_search = GridSearchCV(
#     estimator=random_forest,
#     param_grid=param_grid,
#     cv=5,
#     scoring="f1",
#     verbose=2,
#     n_jobs=1,
# )

# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)
# print(grid_search.best_score_)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)  # zero_division=0 untuk menghindari warning jika tidak ada prediksi positif
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Akurasi: {accuracy:.4f}")
print(f"Presisi: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

dump(random_forest, "model_titanic_klasifikasi.joblib")
