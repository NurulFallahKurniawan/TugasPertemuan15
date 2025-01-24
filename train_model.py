from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

# Data contoh (ganti dengan data Anda)
X = np.array([[5, 150, 80, 30, 200, 25, 1, 45],
              [3, 130, 70, 28, 180, 30, 0, 35],
              [6, 160, 90, 32, 210, 27, 1, 50]])
y = np.array([1, 0, 1])  # Label: 1 = diabetes, 0 = tidak diabetes

# Latih model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Simpan model ke file pickle
with open('knn_pickle', 'wb') as f:
    pickle.dump(model, f)

# Tambahkan print untuk memberi tahu bahwa model sudah dilatih dan disimpan
print("Model telah dilatih dan disimpan ke knn_pickle.")
