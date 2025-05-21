import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Hasil Panen", layout="centered")
st.title("🌾 Prediksi Hasil Panen Tanaman Pangan di Sumatera")
st.markdown("Masukkan data untuk memprediksi hasil panen berdasarkan model Machine Learning.")

# Cek apakah file model tersedia
if os.path.exists("model_xtr.pkl") and os.path.exists("scaler.pkl") and os.path.exists("model_ann.h5"):
    xtr = joblib.load("model_xtr.pkl")
    scaler = joblib.load("scaler.pkl")
    ann_model = load_model("model_ann.h5")
else:
    st.error("❌ File model atau scaler tidak ditemukan.")
    st.stop()

# Input dari user
luas_panen = st.number_input("📐 Luas Panen (ha)", value=1400.0)
curah_hujan = st.number_input("🌧️ Curah Hujan (mm)", value=100.0)
suhu = st.number_input("🌡️ Suhu Rata-rata (°C)", value=27.0)
belanja = st.number_input("💰 Belanja Pertanian (Rp)", value=300000.0)

provinsi = st.selectbox("📍 Provinsi", [
    "Aceh", "Bengkulu", "Jambi", "Kepulauan Bangka Belitung", 
    "Kepulauan Riau", "Lampung", "Riau", "Sumatera Barat", 
    "Sumatera Selatan", "Sumatera Utara"
])

komoditas = st.selectbox("🌱 Komoditas", [
    "Jagung", "Kacang Tanah", "Kedelai", "Ubi Jalar", "Ubi Kayu"
])

# One-hot encoding
provinsi_list = ["Aceh", "Bengkulu", "Jambi", "Kepulauan Bangka Belitung", 
                 "Kepulauan Riau", "Lampung", "Riau", "Sumatera Barat", 
                 "Sumatera Selatan", "Sumatera Utara"]
komoditas_list = ["Jagung", "Kacang Tanah", "Kedelai", "Ubi Jalar", "Ubi Kayu"]

prov_oh = [1 if p == provinsi else 0 for p in provinsi_list]
komoditas_oh = [1 if k == komoditas else 0 for k in komoditas_list]

# Gabungkan fitur
fitur = np.array([[luas_panen, curah_hujan, suhu, belanja] + prov_oh + komoditas_oh])
fitur_scaled = scaler.transform(fitur)

# Prediksi
if st.button("🔍 Prediksi Hasil Panen"):
    with st.spinner("⏳ Memproses prediksi..."):
        # Extra Trees
        pred_xtr = xtr.predict(fitur_scaled)
        pred_xtr = scaler.inverse_transform(pred_xtr.reshape(-1, 1))[0][0]

        # ANN
        pred_ann = ann_model.predict(fitur_scaled.astype('float32'))
        pred_ann = scaler.inverse_transform(pred_ann)[0][0]

    st.success(f"🌳 Prediksi Produksi (Extra Trees): **{pred_xtr:,.2f} Ton**")
    st.success(f"🧠 Prediksi Produksi (ANN): **{pred_ann:,.2f} Ton**")
    st.caption("Model dibuat menggunakan data tanaman pangan di Pulau Sumatera.")
