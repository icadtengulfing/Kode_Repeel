import streamlit as st
import numpy as np
import joblib

# Load pipeline yang berisi scaler + model
pipeline = joblib.load("E:/Machine Learning Docs/diabetes/pipeline_pima_rf_4fitur.pkl")

st.title("Prediksi Diabetes Pasien - Pima Indian (Versi 4 Fitur)")

# Input hanya 4 fitur: Age, BMI, BloodPressure, Pregnancies
preg = st.number_input('Jumlah Kehamilan', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
bp = st.number_input('Tekanan Darah', min_value=0)
age = st.number_input('Umur', min_value=1)

if st.button("Prediksi"):
    # Urutan harus sama dengan model: ['Age', 'BMI', 'BloodPressure', 'Pregnancies']
    data_input = np.array([[age, bmi, bp, preg]])
    
    # Prediksi langsung dengan pipeline (tanpa scaling manual)
    prob = pipeline.predict_proba(data_input)[0][1]
    prediksi = pipeline.predict(data_input)[0]
    
    hasil = "POSITIF Diabetes" if prediksi == 1 else "NEGATIF Diabetes"
    st.write(f"Hasil Prediksi: *{hasil}*")
    st.write(f"Probabilitas Diabetes: *{prob:.2%}*")

# Sidebar penjelasan fitur
st.sidebar.header("Penjelasan Fitur")
st.sidebar.write("""
                 
**Pregnancies**: Jumlah kehamilan yang pernah dialami
                 
**BMI**: Body Mass Index  
- < 18.5: Underweight  
- 18.5–24.9: Normal
- 25–29.9: Overweight  
- ≥ 30: Obese

**BloodPressure**: Tekanan darah sistolik (mmHg)  
- Normal: < 120  
- Elevated: 120–129  
- High: ≥ 130
                 
**Age**: Umur dalam tahun
                 
**Apabila ada kriteria diatas yang sudah melebihi, maka sebaiknya cek langsung ke dokter untuk lebih pasti!**
""")

