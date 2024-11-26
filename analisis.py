import streamlit as st
import pandas as pd


def show_analyst():
    # Path atau URL file CSV
    file_url = "https://drive.google.com/uc?id=1FMmS1amyKNxGOPdNdocpY1Gkt4CMYLql"
    file_url_2 = "https://drive.google.com/uc?id=1tvVfvvZ8SKJtmAeJSIGTRY0VaTGPEEYR"

    # Fungsi untuk memuat data CSV
    @st.cache_data
    def load_data(url, header_row=0, delimiter=';'):
        try:
            df = pd.read_csv(url, delimiter=delimiter,
                             header=header_row, on_bad_lines='skip')
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
            return pd.DataFrame()

    # Fungsi untuk menangani kolom duplikat di df2
    def handle_duplicate_columns(df):
        duplicate_columns = df.columns[df.columns.duplicated()].unique()
        final_columns = {}

        for col in duplicate_columns:
            duplicate_cols_data = df.filter(like=col, axis=1)
            merged_col = duplicate_cols_data.apply(
                lambda row: row.dropna().iloc[0] if not row.dropna().empty else None, axis=1)
            final_columns[col] = merged_col

        unique_columns = df.columns.difference(duplicate_columns)
        final_df = pd.concat(
            [df[unique_columns], pd.DataFrame(final_columns)], axis=1)

        return final_df

    # Load data dari file utama dan file tambahan
    df = load_data(file_url, header_row=0, delimiter=';')
    df2 = load_data(file_url_2, header_row=1, delimiter=',')

    # Normalisasi kolom di df2
    df2.columns = df2.columns.str.strip()
    if 'Nomor NF' not in df2.columns:
        st.error("Kolom 'Nomor NF' tidak ditemukan di df2 setelah normalisasi.")
        return
    df2 = df2.rename(columns={'Nomor NF': 'nonf'})

    # Tangani kolom duplikat di df2
    df2 = handle_duplicate_columns(df2)

    # Menampilkan judul aplikasi
    st.markdown("# 📋 Data Analisis Repeat Buying")

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

        # Dropdown untuk filter 'lb'
        unique_lb = sorted(df['lb'].unique())
        selected_lb = st.selectbox(
            "Filter berdasarkan Lokasi Belajar (lb):", ["Pilih"] + unique_lb)

        if selected_lb != "Pilih":
            # Filter data berdasarkan lokasi belajar (lb)
            filtered_by_lb = df[df['lb'] == selected_lb]

            # Dropdown untuk filter 'jk' khusus tahun 2425
            unique_jk_2425 = sorted(
                filtered_by_lb[filtered_by_lb['idtahun'] == 2425]['jk'].unique())
            selected_jk = st.selectbox(
                "Filter berdasarkan Jenjang Kelas (jk) di Tahun 2425:",
                ["Pilih"] + unique_jk_2425
            )

            # Jika jenjang kelas dipilih
            if selected_jk != "Pilih":
                # Filter data untuk tahun 2425 dan jenjang kelas yang dipilih
                filtered_by_lb_and_jk = filtered_by_lb[
                    (filtered_by_lb['idtahun'] == 2425) & (
                        filtered_by_lb['jk'] == selected_jk)
                ]

                # Pivot data untuk mengubah setiap idtahun menjadi kolom
                pivot_df = filtered_by_lb.pivot_table(
                    index=['nama', 'nonf', 'lb'],
                    columns='idtahun',
                    values='jk',
                    aggfunc='first'
                ).reset_index()

                # Tambahkan kolom tahun ajaran yang hilang secara manual
                all_years = [2021, 2122, 2223, 2324, 2425]
                for year in all_years:
                    if year not in pivot_df.columns:
                        pivot_df[year] = ''

                # Atur ulang urutan kolom
                pivot_df = pivot_df[['nama', 'nonf', 'lb'] + all_years]

                # Hitung `jml_kemunculan` dalam bentuk satuan
                pivot_df['jml_kemunculan'] = pivot_df[all_years].notna().sum(
                    axis=1)

                # Urutkan data berdasarkan `jml_kemunculan` dari tinggi ke rendah
                pivot_df = pivot_df.sort_values(
                    by='jml_kemunculan', ascending=False)

                # Filter tabel akhir berdasarkan jenjang kelas dan tahun ajar 2425
                pivot_df = pivot_df[pivot_df[2425] == selected_jk]

                # Tampilkan tabel data siswa
                with st.container():
                    st.markdown("### 📊 Tabel Data Siswa")
                    st.dataframe(pivot_df)

                # Tabel Data Biaya
                biaya_columns = [
                    'nama', 'nonf', 'idtahun', 'tanggal', 'biaya_formulir', 'biaya_paket', 'biaya_diskon',
                    'jumlah_biaya', 'jumlah_bayar', 'tagihan'
                ]
                biaya_df = df[df['nonf'].isin(pivot_df['nonf'])][biaya_columns]

                # Dropdown untuk filter berdasarkan nama
                selected_name = st.selectbox(
                    "Pilih Nama Siswa untuk Menampilkan Data Biaya:",
                    ["Pilih"] + list(pivot_df['nama'].unique())
                )

                # Jika nama dipilih
                if selected_name != "Pilih":
                    # Filter data biaya untuk nama yang dipilih
                    biaya_by_name = biaya_df[(biaya_df['nama'] == selected_name) & (
                        df['idtahun'].isin([2021, 2122, 2223, 2324, 2425]))]

                    # Tampilkan tabel biaya untuk nama yang dipilih
                    with st.container():
                        st.markdown(
                            f"### 💰 Data Biaya < TA 2425 untuk {selected_name}")
                        st.dataframe(biaya_by_name)

                    # Ambil data tambahan dari df2 berdasarkan 'nonf'
                    selected_nonf = biaya_by_name['nonf'].iloc[0]

                    # Mapping nama kolom panjang ke nama pendek
                    column_aliases = {
                        'Sekolah': 'Sekolah',
                        'Kurikulum yang digunakan di sekolah kamu': 'Kurikulum',
                        'Nomor WA Aktif': 'WA Aktif',
                        'Pada skala 1 sampai 10, seberapa besar kemungkinan kamu mau merekomendasikan BKB NF kepada orang lain?': 'Rating Rekomendasi',
                        'Apa saran/keluhanmu tentang materi pembelajaran yang diberikan BKB NF?': 'Keluhan Materi',
                        'Apa saran/keluhan kamu tentang kakak pengajar BKB NF?': 'Keluhan Pengajar',
                        'Apa saran/keluhan kamu tentang kakak staf admin?': 'Keluhan Admin',
                        'Apa saran/keluhan kamu tentang fasilitas di lokasi belajar BKB NF?': 'Keluhan Fasilitas'
                    }

                    # Filter data tambahan berdasarkan 'nonf'
                    additional_info = df2[df2['nonf'] == selected_nonf][[
                        'Sekolah',
                        'Kurikulum yang digunakan di sekolah kamu',
                        'Nomor WA Aktif',
                        'Pada skala 1 sampai 10, seberapa besar kemungkinan kamu mau merekomendasikan BKB NF kepada orang lain?',
                        'Apa saran/keluhanmu tentang materi pembelajaran yang diberikan BKB NF?',
                        'Apa saran/keluhan kamu tentang kakak pengajar BKB NF?',
                        'Apa saran/keluhan kamu tentang kakak staf admin?',
                        'Apa saran/keluhan kamu tentang fasilitas di lokasi belajar BKB NF?'
                    ]]

                    # Ganti nama kolom sesuai alias jika ada data tambahan
                    if not additional_info.empty:
                        additional_info = additional_info.rename(
                            columns=column_aliases)

                        # Tampilkan data tambahan dalam bentuk tabel
                        with st.container():
                            st.markdown(
                                f"### 🏫 Data Tambahan untuk {selected_name}")
                            # Transpose untuk membuatnya vertikal
                            st.table(additional_info.transpose())
                    else:
                        st.warning(
                            f"⚠️ Tidak ada data tambahan untuk {selected_name}.")

    else:
        st.error(
            "Data tidak dapat dimuat. Pastikan file CSV sesuai format yang diharapkan.")
