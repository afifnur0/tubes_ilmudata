import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# --- 1. KONFIGURASI HALAMAN & TEMA ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

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
    st.error(f"Gagal memuat file model. Pastikan file .pkl tersedia. Error: {e}")
    st.stop()

# --- 3. ANTARMUKA PENGGUNA UTAMA ---
st.markdown("<h1 style='text-align: center; color: #E0E0E0;'>Sistem Prediksi Status Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("---")

metode_pilihan = st.selectbox(
    "Pilih Algoritma AI:",
    options=["Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"]
)

st.markdown("### 📂 Prediksi Massal (Upload File)")
uploaded_file = st.file_uploader("Unggah dataset mahasiswa (CSV/XLSX)", type=["csv", "xlsx"])

def jalankan_prediksi(data_df):
    """Fungsi helper untuk menjalankan prediksi dan memformat output"""
    fitur_scaled = scaler.transform(data_df)
    
    if metode_pilihan == "Decision Tree":
        model_aktif = dt_model
    elif metode_pilihan == "K-Nearest Neighbors":
        model_aktif = knn_model
    else:
        model_aktif = svm_model
        
    hasil = model_aktif.predict(fitur_scaled)
    
    def format_status(x):
        val = str(x).lower()
        if val in ['graduate', '1', '1.0']: return '🎓 GRADUATE'
        elif val in ['dropout', '0', '0.0']: return '❌ DROPOUT'
        return x
        
    return [format_status(x) for x in hasil]

# --- 4. LOGIKA PROSES & DASHBOARD HASIL ---
if uploaded_file is not None:
    # Reset session state jika file baru diunggah agar data tidak tertukar
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
        st.session_state['last_uploaded_file'] = uploaded_file.name
        if 'df_hasil' in st.session_state:
            del st.session_state['df_hasil']

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Validasi Pra-Proses
        if df.shape[1] != 34:
            st.warning(f"File gagal diproses. Sistem mendeteksi {df.shape[1]} kolom. Syarat mutlak adalah 34 kolom fitur tanpa kolom target.", icon="⚠️")
            st.stop()
            
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            st.warning(f"File gagal diproses. Terdeteksi teks pada kolom: {', '.join(non_numeric)}. Pastikan seluruh data berupa angka numerik.", icon="⚠️")
            st.stop()

        if st.button("🤖 Jalankan Prediksi Massal", type="primary", use_container_width=True):
            df_hasil = df.copy()
            df_hasil['Status Final'] = jalankan_prediksi(df)
            
            cols = df_hasil.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df_hasil = df_hasil[cols]
            
            st.session_state['df_hasil'] = df_hasil

        # Menampilkan hasil jika sudah tersimpan di session state (aman dari bug toggle)
        if 'df_hasil' in st.session_state:
            df_hasil = st.session_state['df_hasil']

            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Dashboard Hasil Analisis</h3>", unsafe_allow_html=True)
            
            grad_count = len(df_hasil[df_hasil['Status Final'] == '🎓 GRADUATE'])
            drop_count = len(df_hasil[df_hasil['Status Final'] == '❌ DROPOUT'])
            
            col_metrik, col_chart = st.columns([1, 1.5])
            
            with col_metrik:
                st.metric(label="🎓 Total Graduate", value=grad_count)
                st.metric(label="❌ Total Dropout", value=drop_count)
                
            with col_chart:
                status_df = pd.DataFrame({'Status': ['🎓 GRADUATE', '❌ DROPOUT'], 'Jumlah': [grad_count, drop_count]})
                fig = px.pie(status_df, values='Jumlah', names='Status', hole=0.5, 
                             color='Status', color_discrete_map={
                                 '🎓 GRADUATE':'#2e7b32', '❌ DROPOUT':'#c62828'})
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            tampilkan_detail = st.toggle("Tampilkan Seluruh 34 Kolom Data Mentah")
            
            if tampilkan_detail:
                st.dataframe(df_hasil, use_container_width=True)
            else:
                kolom_esensial = ['Status Final', 'Age at enrollment', 'Scholarship holder', 
                                  'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
                kolom_tersedia = [col for col in kolom_esensial if col in df_hasil.columns]
                st.dataframe(df_hasil[kolom_tersedia], use_container_width=True)

            csv_export = df_hasil.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Prediksi (CSV)", data=csv_export, file_name='hasil_prediksi.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")