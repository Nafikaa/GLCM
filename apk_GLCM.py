import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import joblib
import os

# --- Fungsi Ekstraksi GLCM ---
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
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        prop_vals = graycoprops(glcm, prop)
        features.extend(prop_vals.flatten())
    return np.array(features)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "model_buah.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# --- UI Streamlit ---
st.set_page_config(page_title="Klasifikasi Buah", layout="centered")
st.title("🍍 Klasifikasi Buah Berdasarkan Tekstur Kulit")
st.write("Upload gambar buah (salak, nanas, kiwi), dan sistem akan memprediksi jenis buah berdasarkan tekstur kulitnya menggunakan metode GLCM dan algoritma Random Forest.")

uploaded_file = st.file_uploader("📷 Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    # Proses Gambar
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))

    # Ekstraksi fitur
    features = extract_glcm_features(gray).reshape(1, -1)

    # Prediksi
    prediction = model.predict(features)[0]
    st.success(f"🎯 Prediksi: **{prediction.upper()}**")
