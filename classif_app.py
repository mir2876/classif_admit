import streamlit as st
import pandas as pd
import joblib

# === Load model ===
import os
model = joblib.load(os.path.join(os.path.dirname(__file__), "hasil_rl.pkl"))

# === Min-Max values dari data sebelum scaling ===
feature_min = {
    'motiv_letter_strength': 1,
    'recommendation_strength': 1,
    'gpa': 6.8,
    'research_exp': 0,   # 0 = tidak punya pengalaman riset
    'univ_tier': 0       # 0 = low, 1 = high
}

feature_max = {
    'motiv_letter_strength': 5,
    'recommendation_strength': 5,
    'gpa': 9.92,
    'research_exp': 1,
    'univ_tier': 1
}

# === Fungsi scaling manual ===
def minmax_scale(value, feature):
    min_val = feature_min[feature]
    max_val = feature_max[feature]
    return (value - min_val) / (max_val - min_val)

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Prediksi Penerimaan Mahasiswa Baru Pascasarjana",
    layout="wide",
)

# === Header Utama (selalu muncul di atas) ===##
st.markdown("""
# 🎓 Prediksi Penerimaan Mahasiswa Baru Pascasarjana  
### oleh: Mira Andriyani
---
""")

# === Sidebar Navigasi ===
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio("Pilih:", ["Tentang Aplikasi", "Prediksi"])

# === Halaman 1: Tentang Aplikasi ===
if menu == "Tentang Aplikasi":
    st.subheader("🎯 Tujuan Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk **memprediksi diterima atau tidaknya calon mahasiswa pascasarjana**
    berdasarkan beberapa faktor akademik dan non-akademik, seperti Kekuatan Surat Motivasi, kekuatan
    surat rekomendasi, nilai IPK, pengalaman riset, dan Peringkat Universitas.

    Dengan menggunakan model pembelajaran mesin, aplikasi ini diharapkan dapat membantu calon mahasiswa
    memahami posisi mereka secara lebih obyektif berdasarkan data historis penerimaan. Selain itu, aplikasi ini juga dapat membantu panitia penerimaan mahasiswa dalam menentukan calon mahasiswa yang akan diterima.
    """)

    st.subheader("🧠 Model yang Digunakan")
    st.write("""
    Model yang digunakan dalam aplikasi ini adalah **Regrei Logistik**, 
    Model ini dilatih dengan dataset
    penerimaan mahasiswa pascasarjana luar negeri. Model kemudian disesuaikan agar relevan dengan konteks
    **pendidikan tinggi di Indonesia**, melalui:
    - **Konversi IPK → GPA** agar sesuai dengan skala dataset asal.
    - **Penggantian peringkat universitas (university tier)** dengan **akreditasi universitas**.
    """)

    st.subheader("📊 Dataset Training")
    st.write("""
    Dataset yang digunakan berisi data historis penerimaan mahasiswa pascasarjana,
    termasuk skor akademik dan faktor lain yang relevan.

    Kamu dapat melihat dataset yang digunakan untuk pelatihan model melalui tautan berikut:
    [🔗 Klik untuk melihat dataset](https://drive.google.com/file/d/12Pb4TcNFblNJi_mPhU_YR-5NR2_YzDet/view?usp=sharing)
    """)

    st.subheader("⚙️ Proses Pemodelan")
    st.write("""
    1. **Pra-pemrosesan data:** Membersihkan data (hapus data pencilan, data ganda, data hilang), mengubah tipe data menjadi numerik, dan menormalisasi variabel numerik.  
    2. **Pembagian data:** Data dibagi menjadi data latih dan data uji dengan perbandingan 80:20.
    3. **Mengatasi Multikolinieritas:** Fitur/variabel yang saling berkorelasi dieliminasi.           
    4. **Pelatihan model:** Model Regresi Logistik dilatih untuk meminimalkan kesalahan prediksi.  
    5. **Evaluasi model:** Performa diukur menggunakan presisi, akurasi, rekall, dan f1 score.  
    """)

    st.subheader("📈 Fitur yang Berpengaruh")
    st.write("""
    Berdasarkan hasil pelatihan, dua fitur tereliminasi dan menyisakan beberapa fitur yang berpengaruh terhadap peluang diterima yaitu:
    - Kekuatan Motivation Letter
    - IPK  
    - Kekuatan LOR (Letter of Recommendation)  
    - Pengalaman riset  
    """)

    st.info("Gunakan menu **Prediksi** di sidebar untuk mencoba model ini dengan data Anda sendiri.")

# === Halaman 2: Prediksi ===
elif menu == "Prediksi":
    st.subheader("🧾 Formulir Data Calon Mahasiswa")

    # input di kolom utama
    #gre_score = st.number_input("GRE Score", min_value=0.0, max_value=340.0, value=300.0)
    #toefl_score = st.number_input("TOEFL Score", min_value=0.0, max_value=120.0, value=100.0)
    motiv_letter_strength = st.slider("Kekuatan Surat Motivasi (1–5)", 1, 5, 3)
    recommendation_strength = st.slider("Kekuatan Surat Rekomendasi (1–5)", 1, 5, 3)
    ipk_input = st.number_input("IPK (0.0–4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    # Konversi otomatis IPK → GPA (skala 6.8–9.92)
    gpa = 6.8 + (ipk_input / 4) * (9.92 - 6.8)
    research_exp = st.selectbox("Pengalaman Riset", ["Tidak", "Ya"])
    akreditasi = st.selectbox(
    "Akreditasi Universitas Asal (BAN-PT)",
    ["Unggul / A", "B", "C", "Belum Terakreditasi"]
    )

    # konversi otomatis ke nilai model
    if akreditasi == "Unggul / A" or "B":
        univ_tier_val = 1
    else:
        univ_tier_val = 0
    
    prediksi_btn = st.button("🎯 Prediksi")

    if prediksi_btn:
        # bentuk dataframe input
        # Konversi kategori ke numerik
        research_exp_val = 1 if research_exp == "Ya" else 0
        
        # === Scaling input sesuai model ===
        data_input_scaled = pd.DataFrame({
         'motiv_letter_strength': [minmax_scale(motiv_letter_strength, 'motiv_letter_strength')],
         'recommendation_strength': [minmax_scale(recommendation_strength, 'recommendation_strength')],
         'gpa': [minmax_scale(gpa, 'gpa')],
         'research_exp': [minmax_scale(research_exp_val, 'research_exp')],
         'univ_tier': [minmax_scale(univ_tier_val, 'univ_tier')]
        })
        # prediksi
        pred = model.predict(data_input_scaled)[0]
        pred = float(pred)

        # tampilkan hasil
        st.subheader("📊 Hasil Prediksi")
        if pred >= 0.5:
            st.success("🎉 **berpeluang diterima!**")
        else:
            st.error("❌ **kemungkinan tidak diterima.**")

        st.markdown("""
        ---
        #### 📘 Catatan
        Hasil prediksi menunjukkan estimasi calon mahasiswa akan diterima atau ditolak.
        Model ini tidak menjamin hasil keputusan akhir penerimaan, melainkan hanya memberikan perkiraan berbasis data historis.
        """)

    else:
        st.info("Isi formulir di atas, lalu tekan tombol *Prediksi* untuk melihat hasilnya.")
