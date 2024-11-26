import streamlit as st
import pandas as pd
from PIL import Image
import os
from analisis import show_analyst
from prediksi import show_predict

# Sidebar menu options
menu = st.sidebar.radio(
    "Pilih Menu",
    ("Analisis Jenjang Kelas di Lokasi",
     "Analisis Biaya per Siswa", "Prediksi Repeat Buying")
)

if menu == "Analisis Jenjang Kelas di Lokasi":
    # Path atau URL file CSV
    file_url = "https://drive.google.com/uc?id=1FMmS1amyKNxGOPdNdocpY1Gkt4CMYLql"

    @st.cache_data
    def load_data(url):
        try:
            df = pd.read_csv(url, delimiter=';',
                             quotechar='"', on_bad_lines='skip')
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
            return pd.DataFrame()

    # Load data dari file CSV
    df = load_data(file_url)

    # Menampilkan judul aplikasi
    st.markdown("# 📋 Data Analisis Repeat Buying")

    if not df.empty:
        # Data preprocessing
        df['jk'] = df['jk'].astype(str)
        df = df[df['jk'].str.isalpha()]
        df['lb'] = df['lb'].astype(str).str.strip()
        df['idtahun'] = pd.to_numeric(df['idtahun'], errors='coerce')
        df = df[df['idtahun'].isin([2021, 2122, 2223, 2324, 2425])]
        df = df[['nama', 'nonf', 'jk', 'lb', 'idtahun']]

        # Dropdown untuk filter 'lb'
        unique_lb = sorted(df['lb'].unique())
        selected_lb = st.selectbox(
            "Filter berdasarkan Lokasi Belajar (lb):", ["Pilih"] + unique_lb)

        if selected_lb != "Pilih":
            filtered_by_lb = df[df['lb'] == selected_lb]

            # Pivot data
            pivot_df = filtered_by_lb.pivot_table(
                index=['nama', 'nonf', 'lb'],
                columns='idtahun',
                values='jk',
                aggfunc='first'
            ).reset_index()

            # Menambahkan kolom tahun ajaran yang hilang
            all_years = [2021, 2122, 2223, 2324, 2425]
            for year in all_years:
                if year not in pivot_df.columns:
                    pivot_df[year] = ''

            pivot_df = pivot_df[['nama', 'nonf', 'lb'] + all_years]
            st.markdown("### 📊 Tabel Data Siswa")
            st.dataframe(pivot_df)

            # Dropdown untuk memilih jenjang kelas
            unique_letters = sorted(
                set("".join(filtered_by_lb['jk'].dropna())))
            selected_letter = st.selectbox("Masukkan jenjang kelas yang ingin dihitung:", [
                                           "Pilih"] + unique_letters)

            if selected_letter != "Pilih":
                def count_letter(df, columns, letter):
                    counts = {}
                    rows = {}
                    for col in columns:
                        mask = df[col].apply(
                            lambda x: selected_letter in str(x))
                        rows[col] = df.loc[mask]
                        counts[col] = mask.sum()
                    return counts, rows

                letter_counts, letter_rows = count_letter(
                    pivot_df, all_years, selected_letter)

                st.markdown(
                    f"### 🔢 Jumlah Jenjang Kelas '{selected_letter}' di Tiap Tahun Ajaran")
                result_df = pd.DataFrame.from_dict(
                    letter_counts, orient='index', columns=['Jumlah']).reset_index()
                result_df.rename(
                    columns={'index': 'Tahun Ajaran'}, inplace=True)
                st.dataframe(result_df)

                selected_year = st.selectbox(
                    "Pilih Tahun Ajaran untuk melihat detail siswa:", all_years)

                if selected_year:
                    selected_rows = letter_rows[selected_year]
                    if not selected_rows.empty:
                        st.markdown(
                            f"### 📋 Detail Siswa Tahun Ajaran {selected_year} dengan Huruf '{selected_letter}'")
                        st.dataframe(selected_rows)

elif menu == "Analisis Biaya per Siswa":
    st.title("Analisis Biaya per Siswa")
    show_analyst()

elif menu == "Prediksi Repeat Buying":
    st.title("Prediksi Repeat Buying")
    show_predict()
