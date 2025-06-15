import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# 🚨 WAJIB: Ini HARUS di baris pertama sebelum Streamlit lainnya
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load("model/model.joblib")
    return model

model = load_model()

# --- STREAMLIT UI ---
st.title("🚨 Aplikasi Deteksi Kecurangan Transaksi Keuangan")

st.write("Silakan unggah file CSV berisi data transaksi untuk dilakukan deteksi kecurangan.")

uploaded_file = st.file_uploader("📁 Upload file CSV", type=["csv"])

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Awal")
        st.dataframe(df.head())

        # Siapkan fitur (tanpa isFraud jika ada)
        X = df.drop(columns=['isFraud'], errors='ignore')

        # Prediksi
        y_pred = model.predict(X)
        df['Prediksi'] = y_pred

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df)

        # Evaluasi jika kolom 'isFraud' tersedia
        if 'isFraud' in df.columns:
            st.subheader("📈 Evaluasi Model")
            report = classification_report(df['isFraud'], y_pred, output_dict=True)
            st.json(report)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("📌 Silakan upload file CSV untuk memulai prediksi.")
