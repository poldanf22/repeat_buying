import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def show_predict():
    # Path atau URL file CSV
    file_url = "https://drive.google.com/uc?id=1FMmS1amyKNxGOPdNdocpY1Gkt4CMYLql"

    # Fungsi untuk memuat data CSV
    @st.cache_data
    def load_data(url):
        try:
            df = pd.read_csv(url, delimiter=';',
                             quotechar='"', on_bad_lines='skip')
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
            return pd.DataFrame()  # Kembalikan DataFrame kosong jika ada kesalahan

    # Load data dari file CSV
    df = load_data(file_url)

    st.title("📊 Prediksi Repeat Buying untuk Tahun 2526")

    if not df.empty:
        # Membersihkan data
        df['jk'] = df['jk'].astype(str)
        df = df[df['jk'].str.isalpha()]
        df['lb'] = df['lb'].astype(str).str.strip()
        df = df[df['lb'] != ""]
        df['idtahun'] = pd.to_numeric(df['idtahun'], errors='coerce')
        df = df[df['idtahun'].isin([2021, 2122, 2223, 2324, 2425])]
        df = df[['nama', 'nonf', 'tanggal', 'jk', 'lb', 'idtahun',
                 'biaya_formulir', 'biaya_paket', 'biaya_diskon',
                 'jumlah_biaya', 'jumlah_bayar', 'tagihan']]

    # Preprocessing
    df['idtahun'] = pd.to_numeric(df['idtahun'], errors='coerce')
    df = df[df['idtahun'].isin([2021, 2122, 2223, 2324, 2425])]

    # Dropdown untuk lokasi belajar
    unique_lb = sorted(df['lb'].dropna().unique())
    selected_lb = st.selectbox(
        "Filter berdasarkan Lokasi Belajar (lb):", ["Pilih"] + unique_lb)

    if selected_lb == "Pilih":
        st.info("⚠️ Pilih lokasi belajar untuk melihat hasil prediksi.")
        return

    # Filter data berdasarkan lokasi belajar
    filtered_df = df[df['lb'] == selected_lb]
    if filtered_df.empty:
        st.warning(f"⚠️ Tidak ada data untuk lokasi belajar '{selected_lb}'.")
        return

    # Train Model
    model_file = os.path.join(os.path.dirname(__file__), 'repeat_buying_model.pkl')
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        features = ['jml_kemunculan', 'biaya_formulir', 'biaya_paket',
                    'jumlah_biaya', 'jumlah_bayar', 'tagihan', 'idtahun']
    else:
        model, features = train_model(filtered_df)

    # Feature Importance
    st.markdown("### 📌 Pentingnya Fitur:")
    feature_importance = pd.Series(
        model.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(feature_importance)

    # Prediksi Repeat Buying
    st.markdown("## 🔮 Prediksi Repeat Buying Tahun 2526")
    input_data = {
        'jml_kemunculan': st.number_input("Jumlah Kemunculan Sebelumnya", min_value=0, max_value=10, value=1),
        'biaya_formulir': st.number_input("Biaya Formulir", min_value=0, value=500000),
        'biaya_paket': st.number_input("Biaya Paket", min_value=0, value=1000000),
        'jumlah_biaya': st.number_input("Jumlah Biaya", min_value=0, value=1500000),
        'jumlah_bayar': st.number_input("Jumlah Bayar", min_value=0, value=1400000),
        'tagihan': st.number_input("Tagihan", min_value=0, value=100000),
        'idtahun': 2526
    }

    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(
                f"✅ Siswa diprediksi akan melakukan repeat buying dengan probabilitas {probability * 100:.2f}%.")
        else:
            st.warning(
                f"❌ Siswa tidak diprediksi melakukan repeat buying dengan probabilitas {(1 - probability) * 100:.2f}%.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")


def train_model(data):
    data['jml_kemunculan'] = data.groupby('nonf')['idtahun'].transform('count')
    data['repeat_buying'] = (data['jml_kemunculan'] > 1).astype(int)

    features = ['jml_kemunculan', 'biaya_formulir', 'biaya_paket', 'jumlah_biaya',
                'jumlah_bayar', 'tagihan', 'idtahun']
    target = 'repeat_buying'

    data = data[features + [target]].dropna()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'repeat_buying_model.pkl')
    return model, features
