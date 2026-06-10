import streamlit as st
import numpy as np
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", page_icon="🎓")

# --- 2. LOAD MODEL & SCALER ---
# st.cache_resource digunakan agar model tidak diload berulang kali setiap klik tombol
@st.cache_resource
def load_models():
    # Pastikan keempat file ini ada di folder yang sama dengan app.py
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

# Panduan Pengujian (Menggantikan Alert Vue)
with st.expander("ℹ️ Panduan Pengujian (Testing)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **✅ Skenario Lulus (Graduate)**
        * Usia: **18 - 20**
        * Nilai Semester 1 & 2: **13.0 - 20.0**
        * *AI mendeteksi mahasiswa muda, reguler, mendapat beasiswa, dan SKS lulus penuh.*
        """)
    with col2:
        st.markdown("""
        **❌ Skenario Putus Studi (Dropout) Mutlak**
        * Usia: Bebas
        * Nilai Semester 1 & 2: **0.0 - 5.0**
        * *Nilai anjlok memicu sistem menganggap mahasiswa menunggak SPP & SKS 0.*
        """)
    st.markdown("""
    **⚠️ Skenario Kritis (Pengaruh Usia)**
    * Gunakan Nilai Pas-pasan: **11.5** untuk kedua semester.
    * Uji Usia **19** ➔ AI menebak **Graduate / Enrolled**.
    * Uji Usia **30** ➔ AI menebak **Dropout**.
    * *Logika AI: Mahasiswa berumur dengan nilai pas-pasan diasumsikan sebagai pekerja kelas malam yang kesulitan finansial & hanya lulus separuh SKS.*
    """)

# Pilihan Model & Input Data
metode_pilihan = st.selectbox(
    "Pilih Otak AI (Algoritma):",
    options=["Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"]
)

usia = st.number_input("Usia saat mendaftar (Age at enrollment):", min_value=15, max_value=60, value=19, step=1)

col_n1, col_n2 = st.columns(2)
with col_n1:
    n1 = st.number_input("Nilai Sem 1 (Skala 0 - 20):", min_value=0.0, max_value=20.0, value=13.5, step=0.1)
with col_n2:
    n2 = st.number_input("Nilai Sem 2 (Skala 0 - 20):", min_value=0.0, max_value=20.0, value=14.0, step=0.1)

# --- 4. LOGIKA PREDIKSI (Menggantikan Backend API/Javascript) ---
if st.button("🤖 Jalankan Prediksi Sekarang", use_container_width=True):
    
    # Replikasi array default dari Vue
    data_lengkap = [
        1, 1, 1, 5, 1, 1, 1, 22, 14, 10, 5, 0, 0, 0, 1, 1, 1, 
        20, 0, 1, 6, 7, 6, 13.5, 0, 1, 6, 9, 6, 13.5, 0, 10.8, 1.4, 1.74 
    ]
    
    # Timpa fitur berdasarkan inputan UI
    data_lengkap[17] = usia
    data_lengkap[23] = n1
    data_lengkap[29] = n2
    
    # Replikasi logika skenario (IF/ELSE Javascript ke Python)
    if n1 <= 5 and n2 <= 5:
        data_lengkap[22] = 0 
        data_lengkap[28] = 0 
        data_lengkap[14] = 0 
        data_lengkap[13] = 1 
        data_lengkap[16] = 0 
    elif n1 <= 12.5 and n2 <= 12.5 and usia >= 25:
        data_lengkap[22] = 3 
        data_lengkap[28] = 3 
        data_lengkap[14] = 0 
        data_lengkap[4]  = 0 
        data_lengkap[16] = 0 
    elif n1 <= 12.5 and n2 <= 12.5 and usia < 25:
        data_lengkap[22] = 5 
        data_lengkap[28] = 5 
        data_lengkap[14] = 1 
        data_lengkap[4]  = 1 
        data_lengkap[16] = 1 
    else:
        data_lengkap[22] = 6 
        data_lengkap[28] = 6 
        data_lengkap[14] = 1 
        if usia >= 25:
            data_lengkap[4]  = 0 
            data_lengkap[16] = 0 
        else:
            data_lengkap[4]  = 1 
            data_lengkap[16] = 1 

    # Eksekusi Machine Learning
    try:
        # Ubah ke numpy array 2D
        fitur_mahasiswa = np.array([data_lengkap])
        
        # Scaling fitur
        fitur_scaled = scaler.transform(fitur_mahasiswa)
        
        # Pilih algoritma berdasarkan dropdown
        if metode_pilihan == "Decision Tree":
            model_aktif = dt_model
        elif metode_pilihan == "K-Nearest Neighbors":
            model_aktif = knn_model
        else:
            model_aktif = svm_model
            
        # Lakukan tebakan
        hasil_prediksi = model_aktif.predict(fitur_scaled)
        hasil_final = hasil_prediksi[0]
        
        # Output Visual (Menggantikan CSS v-if di Vue)
        st.markdown("---")
        st.write("##### Hasil Analisis AI:")
        
        # Penyesuaian output berdasarkan label model
        if str(hasil_final).lower() in ['graduate', '1', '1.0']:
            st.success("🎓 GRADUATE (LULUS)")
        elif str(hasil_final).lower() in ['dropout', '0', '0.0']:
            st.error("❌ DROPOUT (PUTUS STUDI)")
        elif str(hasil_final).lower() in ['enrolled', '2', '2.0']:
            st.warning("⏳ ENROLLED (MASIH AKTIF)")
        else:
            st.info(f"HASIL: {hasil_final}")
            
        st.caption(f"Algoritma yang digunakan: {metode_pilihan}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses Machine Learning: {e}")