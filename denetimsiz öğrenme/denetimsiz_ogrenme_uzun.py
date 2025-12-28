from __future__ import annotations


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

"""
Basit bir denetimsiz öğrenme (unsupervised learning) örneği.

Bu betik, Iris veri seti üzerinde KMeans kümeleme uygular:
- Özellikleri ölçekler (StandardScaler)
- İki bileşenli PCA ile görselleştirilebilir uzaya indirger
- KMeans ile 3 küme bulur ve Silhouette skoru raporlar
- (Varsa) matplotlib ile kümeleri 2B grafikte gösterir

Not: Eğer `scikit-learn` veya `matplotlib` sisteminizde kurulu değilse,
betik bunları nazikçe bildirir ve görselleştirme adımını atlar.
"""

import sys
from typing import Optional

try:
	from sklearn import datasets
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score
except Exception as exc:
	print("Gerekli paketlerden bazıları eksik görünüyor: scikit-learn.")
	print("Lütfen aşağıdaki komutla yükleyin ve yeniden deneyin:")
	print("\n    pip install scikit-learn\n")
	sys.exit(1)


def try_import_matplotlib() -> Optional[object]:
	"""Matplotlib'i opsiyonel olarak yüklemeyi dener; yoksa None döner."""
	try:
		import matplotlib.pyplot as plt  # type: ignore
		return plt
	except Exception:
		return None


def kmeans_on_iris(random_state: int = 42) -> dict:
	"""
	Iris veri setinde KMeans uygulayın ve temel çıktıları döndürün.

	Dönen sözlük:
	- X: Ölçeklenmiş orijinal özellikler (n,4)
	- X_pca: PCA ile 2 bileşene indirgenmiş özellikler (n,2)
	- y_true: Gerçek sınıf etiketleri (0,1,2)
	- labels: KMeans tahmini küme etiketleri
	- silhouette: Silhouette skoru (yüksekse daha iyi ayrışmış kümeler)
	- kmeans: Eğitilmiş KMeans nesnesi
	"""
	iris = datasets.load_iris()
	X = iris.data
	y_true = iris.target

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=2, random_state=random_state)
	X_pca = pca.fit_transform(X_scaled)

	# scikit-learn sürümleri arası uyum için n_init'i sabitleyin
	kmeans = KMeans(n_clusters=3, n_init=10, random_state=random_state)
	labels = kmeans.fit_predict(X_scaled)

	silhouette = silhouette_score(X_scaled, labels)

	return {
		"X": X_scaled,
		"X_pca": X_pca,
		"y_true": y_true,
		"labels": labels,
		"silhouette": silhouette,
		"kmeans": kmeans,
	}


def plot_clusters(X_pca, labels, y_true) -> None:
	"""Kümeleri PCA-2B uzayında görselleştir (matplotlib varsa)."""
	plt = try_import_matplotlib()
	if plt is None:
		print("Matplotlib bulunamadı; görselleştirme adımı atlandı.")
		print("İsterseniz yüklemek için: pip install matplotlib")
		return

	fig, ax = plt.subplots(figsize=(7, 5))
	sc = ax.scatter(
		X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=60, alpha=0.8
	)
	ax.set_title("Iris - KMeans Kümeleri (PCA ile 2B)")
	ax.set_xlabel("PCA Bileşen 1")
	ax.set_ylabel("PCA Bileşen 2")
	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("Küme Etiketi")

	# Karşılaştırma için gerçek sınıfları nokta kenarı rengiyle gösterin
	# (yalnızca görsel fikir vermesi için)
	try:
		import numpy as np  # opsiyonel
		edge_colors = np.array(["black", "white", "grey"])
		edge = edge_colors[(y_true % len(edge_colors))]
		for i in range(X_pca.shape[0]):
			ax.scatter(X_pca[i, 0], X_pca[i, 1], s=60, edgecolors=edge[i], facecolors='none')
		ax.legend([
			"Dolgulu renk: KMeans kümesi",
			"Siyah/Beyaz/Gri kenar: Gerçek sınıf (referans)"
		], loc="best")
	except Exception:
		pass

	plt.tight_layout()
	plt.show()


def main() -> None:
	print("Iris veri setinde KMeans kümeleme örneği çalışıyor...\n")
	results = kmeans_on_iris()
	silhouette = results["silhouette"]
	labels = results["labels"]
	X_pca = results["X_pca"]
	y_true = results["y_true"]

	# Temel özet
	import numpy as np
	unique, counts = np.unique(labels, return_counts=True)
	print("Küme dağılımı (etiket: adet):", dict(zip(unique.tolist(), counts.tolist())))
	print(f"Silhouette skoru: {silhouette:.3f}")

	# (Opsiyonel) görselleştir
	plot_clusters(X_pca, labels, y_true)


if __name__ == "__main__":
	main()

