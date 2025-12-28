# denetimli öğrenme modeli kullanarak öğrenci başarı notu tahmini


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
# Veri setini oluşturma
sinav1 = np.random.randint(50, 100, size=100)
sinav2 = np.random.randint(50, 100, size=100)
# Başarı notu sınav notlarına bağlı olarak hesaplanıyor (ağırlıklı ortalama + gürültü)
basari_notu = (sinav1 * 0.4 + sinav2 * 0.6 + np.random.normal(0, 5, 100)).clip(0, 100).astype(int)

data = {
    'ogrenci_id': range(1, 101),
    'sinav1': sinav1,
    'sinav2': sinav2,
    'basari_notu': basari_notu
}

df = pd.DataFrame(data)

# Özellikler ve hedef değişkenin belirlenmesi
X = df[['sinav1', 'sinav2']]
y = df['basari_notu']
# Veri setinin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Modelin oluşturulması ve eğitilmesi
model = LinearRegression(
)
model.fit(X_train, y_train)
# Test verisi üzerinde tahmin yapılması
y_pred = model.predict(X_test)
# Model performansının değerlendirilmesi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
# Yeni bir öğrenci için başarı notu tahmini
new_student = pd.DataFrame([[85, 90]], columns=['sinav1', 'sinav2'])  # sinav1=85, sinav2=90
predicted_score = model.predict(new_student)
print(f"Tahmin Edilen Başarı Notu: {predicted_score[0]}")