import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", page_icon="🎓")

# --- 2. LOAD MODEL & SCALER ---
@st.cache_resource
def load_models():
    dt_model = joblib.load('decision_tree_hpo_terbaik.pkl')
    knn_model = joblib.load('k-nn_terbaik.pkl')
    svm_model = joblib.load('model_SVM.pkl')
    scaler = joblib.load('scaler.pkl')
    return dt_model, knn_model, svm_model, scaler

try:
    dt_model, knn_model, svm_model, scaler = load_models()
except Exception as e:
    st.error(f"Gagal memuat file model. Pastikan file .pkl sudah diekspor ke folder ini. Error: {e}")
    st.stop()

# --- 3. ANTARMUKA PENGGUNA (UI / Frontend) ---
st.markdown("<h2 style='text-align: center;'>Sistem Prediksi Status Mahasiswa</h2>", unsafe_allow_html=True)

# Panduan Pengujian File (Diperbarui)
with st.expander("ℹ️ Panduan Persiapan File Pengujian", expanded=True):
    st.markdown("""
    **Format File Wajib:**
    * File Excel (.xlsx) atau CSV (.csv).
    * Berisi tepat **34 kolom berurutan** tanpa kolom target/label.
    * Baris pertama adalah *header* (nama kolom).
    
    **Cara Menguji Skenario AI (Ubah data di file Excel Anda sebelum diunggah):**
    Di dalam dataset Anda, asumsikan:
    * Kolom ke-18 = Usia saat mendaftar
    * Kolom ke-24 = Nilai Semester 1
    * Kolom ke-30 = Nilai Semester 2
    
    ✅ **Uji Skenario Lulus:** Buat satu baris mahasiswa dengan nilai Sem 1 & 2 di atas `13.0` dan Usia `18 - 20`.
    ❌ **Uji Skenario Dropout:** Buat satu baris mahasiswa dengan nilai Sem 1 & 2 di bawah `5.0`.
    ⚠️ **Uji Skenario Usia Kritis:** Buat dua baris dengan nilai Sem 1 & 2 pas-pasan (misal: `11.5`). Pada baris pertama isi usia `19` (Sistem akan menebak Graduate/Enrolled). Pada baris kedua isi usia `30` (Sistem akan berubah menebak Dropout).
    """)

# Pilihan Model
metode_pilihan = st.selectbox(
    "Pilih Otak AI (Algoritma):",
    options=["Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"]
)

# Area Drag & Drop File
st.write("### Upload Data Mahasiswa")
uploaded_file = st.file_uploader("Unggah dataset untuk diproses secara massal", type=["csv", "xlsx"])

# --- 4. LOGIKA PREDIKSI ---
if uploaded_file is not None:
    try:
        # Membaca file menjadi DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("Preview Data yang Diunggah:")
        st.dataframe(df.head())

        if st.button("🤖 Jalankan Prediksi Massal", use_container_width=True):
            try:
                # Proses Scaling fitur menggunakan scaler.pkl
                fitur_scaled = scaler.transform(df)
                
                # Memilih model
                if metode_pilihan == "Decision Tree":
                    model_aktif = dt_model
                elif metode_pilihan == "K-Nearest Neighbors":
                    model_aktif = knn_model
                else:
                    model_aktif = svm_model
                    
                # Eksekusi prediksi massal
                hasil_prediksi = model_aktif.predict(fitur_scaled)
                
                def format_status(x):
                    val = str(x).lower()
                    if val in ['graduate', '1', '1.0']: return '🎓 GRADUATE'
                    elif val in ['dropout', '0', '0.0']: return '❌ DROPOUT'
                    elif val in ['enrolled', '2', '2.0']: return '⏳ ENROLLED'
                    return x
                
                # Memasukkan hasil prediksi ke dalam salinan DataFrame
                df_hasil = df.copy()
                df_hasil['Output Model'] = hasil_prediksi
                df_hasil['Status Final'] = df_hasil['Output Model'].apply(format_status)

                st.markdown("---")
                st.write(f"##### Hasil Analisis Keseluruhan ({metode_pilihan}):")
                
                # Menampilkan tabel hasil
                st.dataframe(df_hasil)

            except ValueError as ve:
                st.error(f"Error bentuk data: {ve}. Pastikan jumlah kolom pada file persis 34 dan hanya berisi angka numerik tanpa teks.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses Machine Learning: {e}")

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")