import streamlit as st
from logic import *  
st.title('Aplikasi Prediksi Persetujuan Pinjaman')
st.write("Aplikasi ini memprediksi apakah pinjaman akan disetujui berdasarkan input dari pengguna.")
st.header('Masukkan Data Calon Peminjam')

data = {
    'gender': st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan']),
    'married': st.selectbox('Status Pernikahan', ['Ya', 'Tidak']),
    'dependents': st.selectbox('Jumlah Tanggungan', ['0', '1', '2', '3+']),
    'education': st.selectbox('Pendidikan', ['Sarjana', 'Tidak Sarjana']),
    'self_employed': st.selectbox('Wiraswasta', ['Ya', 'Tidak']),
    'applicant_income': st.number_input('Pendapatan Pemohon', min_value=0),
    'coapplicant_income': st.number_input('Pendapatan Pasangan', min_value=0),
    'loan_amount': st.number_input('Jumlah Pinjaman (dalam ribuan)', min_value=0.0),
    'loan_amount_term': st.selectbox('Jangka Waktu Pinjaman (bulan)', [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0]),
    'credit_history': st.selectbox('Riwayat Kredit (1 jika ada, 0 jika tidak)', [0.0, 1.0]),
    'property_area': st.selectbox('Area Properti', ['Pedesaan', 'Semiurban', 'Perkotaan']),
}

if st.button('Prediksi Persetujuan Pinjaman'):
    try:
        prediction, probability, _ = predict_loan_approval(data)
        st.header('Hasil Prediksi')
        if prediction == 1:
            st.success('Pinjaman DIPREDIKSI DISETUJUI')
        else:
            st.error('Pinjaman DIPREDIKSI DITOLAK')
        st.write(f"Probabilitas Persetujuan: {probability:.2f}")

        if probability > 0.7:
            save_to_database(data, 'Disetujui' if prediction == 1 else 'Ditolak', probability)
            st.info("Hasil prediksi disimpan ke database.")
        else:
            st.warning("Probabilitas rendah, data tidak disimpan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
