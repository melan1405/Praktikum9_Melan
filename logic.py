import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error
import joblib

# Load model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model file tidak ditemukan.")

# Fungsi membuat koneksi ke database
def create_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='nama_database'  # GANTI dengan nama database kamu
        )
        return conn
    except Error as e:
        raise ConnectionError(f"Koneksi gagal: {e}")

# Fungsi prediksi
def predict_loan_approval(data: dict):
    input_df = pd.DataFrame([{
        'Gender': 1 if data['gender'] == 'Laki-laki' else 2,
        'Married': 1 if data['married'] == 'Ya' else 2,
        'Dependents': 3 if data['dependents'] == '3+' else int(data['dependents']),
        'Education': 1 if data['education'] == 'Sarjana' else 2,
        'Self_Employed': 1 if data['self_employed'] == 'Ya' else 2,
        'ApplicantIncome': data['applicant_income'],
        'CoapplicantIncome': data['coapplicant_income'],
        'LoanAmount': data['loan_amount'],
        'Loan_Amount_Term': data['loan_amount_term'],
        'Credit_History': data['credit_history'],
        'Property_Area': 1 if data['property_area'] == 'Pedesaan' else (2 if data['property_area'] == 'Semiurban' else 3)
    }])

    # Reorder kolom
    input_df = input_df[['Gender', 'Married', 'Education', 'Self_Employed',
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                         'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability, input_df

# Simpan hasil prediksi ke database
def save_to_database(data: dict, prediction: str, probability: float):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area,
            prediction, probability
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data['gender'], data['married'], data['dependents'], data['education'],
        data['self_employed'], data['applicant_income'], data['coapplicant_income'],
        data['loan_amount'], data['loan_amount_term'], data['credit_history'],
        data['property_area'], prediction, probability
    ))
    conn.commit()
    cursor.close()
    conn.close()

# Ambil semua data prediksi
def fetch_all_predictions():
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    cursor.close()
    conn.close()
    return df

# Streamlit tampilan
st.markdown("---")
st.subheader("📊 Data Hasil Prediksi yang Tersimpan")

if st.button("Tampilkan Data dari Database"):
    try:
        df = fetch_all_predictions()
        if df.empty:
            st.info("Belum ada data yang disimpan.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Gagal menampilkan data: {e}")
