import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('./data/diabetes.csv')


X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

nama_fitur = model.feature_names_in_
fitur_penting = model.feature_importances_
df_baru = pd.Series(fitur_penting, index=nama_fitur)
sort_feature = df_baru.sort_values(ascending=False)

fitur_terpilih = sort_feature.head(5).index.to_list()
print(fitur_terpilih)

X_train_baru = X_train[fitur_terpilih]
X_test_baru = X_test[fitur_terpilih] 

model.fit(X_train_baru, y_train)

y_pred_baru = model.predict(X_test_baru)
evaluasi = classification_report(y_test, y_pred_baru)
print(evaluasi)



















# y_pred = model.predict(X_test)
# evaluasi = classification_report(y_test, y_pred)
# print(evaluasi)

# # Korelasi menggunakan heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("Heatmap Korelasi")
# plt.show()