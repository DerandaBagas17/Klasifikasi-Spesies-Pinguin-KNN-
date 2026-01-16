
import streamlit as st
import pandas as pd
import joblib
import os

# --- Muat dataset lagi di dalam app.py untuk mendapatkan min/max/mean untuk slider ---
# Dalam aplikasi produksi yang sebenarnya, nilai min/max/mean ini mungkin disimpan secara terpisah
# atau dihitung sebelumnya dan di-*hardcode* untuk kinerja.
try:
    app_df = pd.read_csv('penguins_size.csv')
except FileNotFoundError:
    st.error("Error: 'penguins_size.csv' tidak ditemukan. Pastikan file dataset berada di direktori yang sama dengan app.py.")
    st.stop()

# Tangani nilai yang hilang untuk app_df seperti yang dilakukan di notebook
for col in ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    app_df[col] = app_df[col].fillna(app_df[col].mean())
app_df['sex'] = app_df['sex'].fillna(app_df['sex'].mode()[0])

# --- Muat model KNN yang terlatih, LabelEncoder, StandardScaler, dan nama kolom fitur ---
try:
    knn_model = joblib.load('best_knn_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib') # Muat X.columns yang disimpan
except FileNotFoundError:
    st.error("Error: File model atau komponen pra-pemrosesan tidak ditemukan. Pastikan 'best_knn_model.joblib', 'label_encoder.joblib', 'scaler.joblib', dan 'feature_columns.joblib' berada di direktori yang sama dengan app.py.")
    st.stop()

st.title('Aplikasi Prediksi Spesies Pinguin')
st.write('Masukkan karakteristik fisik pinguin untuk memprediksi spesiesnya.')

st.sidebar.header('Fitur Pinguin')

def user_input_features():
    # Gunakan app_df untuk rentang slider
    culmen_length_mm = st.sidebar.slider('Panjang Culmen (mm)', float(app_df['culmen_length_mm'].min()), float(app_df['culmen_length_mm'].max()), float(app_df['culmen_length_mm'].mean()))
    culmen_depth_mm = st.sidebar.slider('Kedalaman Culmen (mm)', float(app_df['culmen_depth_mm'].min()), float(app_df['culmen_depth_mm'].max()), float(app_df['culmen_depth_mm'].mean()))
    flipper_length_mm = st.sidebar.slider('Panjang Sirip (mm)', float(app_df['flipper_length_mm'].min()), float(app_df['flipper_length_mm'].max()), float(app_df['flipper_length_mm'].mean()))
    body_mass_g = st.sidebar.slider('Massa Tubuh (g)', float(app_df['body_mass_g'].min()), float(app_df['body_mass_g'].max()), float(app_df['body_mass_g'].mean()))

    island = st.sidebar.selectbox('Pulau', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Jenis Kelamin', ('MALE', 'FEMALE'))

    data = {
        'culmen_length_mm': culmen_length_mm,
        'culmen_depth_mm': culmen_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'island_Dream': 1 if island == 'Dream' else 0,
        'island_Torgersen': 1 if island == 'Torgersen' else 0,
        'sex_FEMALE': 1 if sex == 'FEMALE' else 0,
        'sex_MALE': 1 if sex == 'MALE' else 0 
    }
    features = pd.DataFrame(data, index=[0])

    # Pastikan urutan kolom sesuai dengan data pelatihan X.columns menggunakan feature_columns yang dimuat
    features = features[feature_columns] 
    return features

input_df = user_input_features()

st.subheader('Fitur Input Pengguna')
st.write(input_df)

if st.button('Prediksi'):
    # Buat salinan untuk menghindari modifikasi input_df asli secara langsung
    input_df_scaled = input_df.copy()

    # Identifikasi fitur numerik (berdasarkan pra-pemrosesan notebook)
    numerical_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Terapkan scaler ke kolom numerik dari input_df
    input_df_scaled[numerical_features] = scaler.transform(input_df_scaled[numerical_features])

    # Buat prediksi
    prediction = knn_model.predict(input_df_scaled)

    # Dekode prediksi
    predicted_species = label_encoder.inverse_transform(prediction)

    st.subheader('Prediksi')
    st.write(f'Spesies yang diprediksi adalah: **{predicted_species[0]}**')

    # --- Tambahkan tampilan gambar berdasarkan spesies yang diprediksi ---
    # FIXED: Gunakan jalur relatif untuk gambar
    image_path = f'{predicted_species[0]}.jpg' # Diasumsikan gambar berada di direktori yang sama dengan app.py
    if os.path.exists(image_path):
        st.image(image_path, caption=predicted_species[0], width=200)
    else:
        st.warning(f"Gambar untuk {predicted_species[0]} tidak ditemukan di {image_path}. Pastikan file gambar ada di direktori yang sama dengan app.py.")

