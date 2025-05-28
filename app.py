import streamlit as st
import pandas as pd
import joblib

# Load model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Pastikan 'logistic_regression_model.pkl' ada di direktori yang sama.")
    st.stop()

# Judul Aplikasi
st.title('Aplikasi Prediksi Persetujuan Pinjaman')
st.write("""
Aplikasi ini memprediksi apakah pinjaman akan disetujui berdasarkan input
dari pengguna.
""")

# Input Form
st.header('Masukkan Data Calon Peminjam')
gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
married = st.selectbox('Status Pernikahan', ['Ya', 'Tidak'])
education = st.selectbox('Pendidikan', ['Sarjana', 'Tidak Sarjana'])
self_employed = st.selectbox('Wiraswasta', ['Ya', 'Tidak'])
applicant_income = st.number_input('Pendapatan Pemohon', min_value=0)
coapplicant_income = st.number_input('Pendapatan Pasangan', min_value=0)
loan_amount = st.number_input('Jumlah Pinjaman (dalam ribuan)', min_value=0.0)
loan_amount_term = st.selectbox('Jangka Waktu Pinjaman (dalam bulan)', [
    12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0])
credit_history = st.selectbox('Riwayat Kredit (1 jika ada, 0 jika tidak)', [0.0, 1.0])
property_area = st.selectbox('Area Properti', ['Pedesaan', 'Semiurban', 'Perkotaan'])

# Prediksi
if st.button('Prediksi Persetujuan Pinjaman'):
    # Buat DataFrame input sesuai fitur saat training
    input_data = {
        'Gender': [1 if gender == 'Laki-laki' else 2],
        'Married': [1 if married == 'Ya' else 2],
        'Education': [1 if education == 'Sarjana' else 2],
        'Self_Employed': [1 if self_employed == 'Ya' else 2],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [1 if property_area == 'Pedesaan' else (2 if property_area == 'Semiurban' else 3)]
    }

    input_df = pd.DataFrame(input_data)

    try:
        # Pastikan hanya menggunakan fitur yang dikenal model
        input_df = input_df[model.feature_names_in_]
    except AttributeError:
        st.warning("Model tidak memiliki atribut 'feature_names_in_'. Pastikan urutan kolom input sesuai saat pelatihan.")

    # Lakukan prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    # Tampilkan hasil prediksi
    st.header('Hasil Prediksi')
    if prediction[0] == 1:
        st.success('Pinjaman DIPREDIKSI DISETUJUI')
    else:
        st.error('Pinjaman DIPREDIKSI DITOLAK')
    st.write(f"Probabilitas Persetujuan: {prediction_proba[0]:.2f}")
