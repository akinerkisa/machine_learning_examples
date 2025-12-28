"""
Basit denetimsiz öğrenme: Öğrencileri sınav notlarına göre KMeans ile kümeler.

Bu örnek, iki sınav notundan oluşan sentetik bir veri seti oluşturur ve
KMeans ile 3 küme (ör. düşük/orta/yüksek performans) bulur. Küme dağılımını,
küme merkezlerini ve yeni bir öğrencinin hangi kümeye düştüğünü raporlar.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main() -> None:
	np.random.seed(42)

	# Sentetik veri: 100 öğrenci, 50-100 arası iki sınav notu
	sinav1 = np.random.randint(50, 100, size=100)
	sinav2 = np.random.randint(50, 100, size=100)

	df = pd.DataFrame({
		"ogrenci_id": np.arange(1, 101),
		"sinav1": sinav1,
		"sinav2": sinav2,
	})

	# Özellik matrisi
	X = df[["sinav1", "sinav2"]].values

	# KMeans ile 3 küme bul
	kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
	labels = kmeans.fit_predict(X)
	df["kume"] = labels

	# Küme merkezleri ve dağılımı
	centers = kmeans.cluster_centers_
	unique, counts = np.unique(labels, return_counts=True)
	sil = silhouette_score(X, labels)

	print("Küme dağılımı (etiket: adet):", dict(zip(unique.tolist(), counts.tolist())))
	print("Küme merkezleri (sinav1, sinav2):")
	for i, c in enumerate(centers):
		print(f"  Küme {i}: ({c[0]:.2f}, {c[1]:.2f})")
	print(f"Silhouette skoru: {sil:.3f}")

	# Yeni bir öğrenci için küme tahmini
	new_student = pd.DataFrame([[85, 90]], columns=["sinav1", "sinav2"])  # sinav1=85, sinav2=90
	predicted_cluster = kmeans.predict(new_student.values)[0]
	print(f"Yeni öğrencinin tahmini kümesi: {predicted_cluster}")

	# İsterseniz sonuçları CSV olarak kaydedebilirsiniz
	# df.to_csv("ogrenci_kumeleme_sonuclari.csv", index=False)

if __name__ == "__main__":
	main()
