import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Ekstraksi Fitur GLCM ---
def extract_glcm_features(gray_image):
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        prop_vals = graycoprops(glcm, prop)
        features.extend(prop_vals.flatten())
    return features

# --- Load Dataset ---
def load_dataset(dataset_path):
    features, labels = [], []

    if not os.path.exists(dataset_path):
        print(f"❌ Folder dataset tidak ditemukan: {dataset_path}")
        exit()

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            try:
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (128, 128))
                feat = extract_glcm_features(gray)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"❌ Gagal memproses {img_path}: {e}")
                continue

    return np.array(features), np.array(labels)

# --- MAIN ---
if __name__ == "__main__":
    dataset_path = "dataset"  # folder dataset harus berada di folder yang sama
    print("📂 Memuat dataset dari:", os.path.abspath(dataset_path))

    X, y = load_dataset(dataset_path)

    print("🧪 Split data latih dan uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Melatih model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("📈 Evaluasi model...")
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model_buah.pkl")
    print("✅ Model disimpan sebagai 'model_buah.pkl'")
