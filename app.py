import streamlit as st
import pandas as pd
import numpy as np
import joblib

from streamlit_option_menu import option_menu
from sklearn.metrics import silhouette_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Clustering Penggunaan Aplikasi Digital",
    page_icon="📱",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>
        Clustering Penggunaan Aplikasi Digital Mahasiswa
    </h1>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
try:
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan kmeans_model.pkl dan scaler.pkl ada di folder yang sama dengan app.py")
    st.stop()

# =====================================================
# LOAD DATASET (dari file lokal, tanpa upload)
# =====================================================
@st.cache_data
def load_dataset():
    file_path = "dataset.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File dataset '{file_path}' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
        st.stop()

df_raw = load_dataset()

# =====================================================
# RENAME KOLOM (sesuai dengan form asli)
# =====================================================
df = df_raw.copy()
df = df.rename(columns={
    "Seberapa sering Anda menggunakan aplikasi media sosial (Instagram, TikTok, dll)?  ": "sosial_media",
    "Seberapa sering Anda menggunakan aplikasi chatting (WhatsApp, Telegram, dll)  ": "chatting",
    "Seberapa sering Anda menggunakan aplikasi streaming video (YouTube, Netflix, dll)?  ": "streaming_video",
    "Seberapa sering Anda menggunakan aplikasi musik (Spotify, Joox, dll)?  ": "musik",
    "Seberapa sering Anda menggunakan aplikasi game mobile?  ": "game_mobile",
    "Seberapa sering Anda menggunakan aplikasi e-commerce (Shopee, Tokopedia, dll)?  ": "ecommerce",
    "Seberapa sering Anda menggunakan aplikasi transportasi online (Gojek, Grab, dll)?  ": "transportasi_online"
})

# =====================================================
# DURASI MAPPING
# =====================================================
durasi_mapping = {
    "< 1 jam": 1,
    "1–3 jam": 2,
    "3–5 jam": 3,
    "5–7 jam": 4,
    "> 7 jam": 5
}

durasi_col = "Berapa lama Anda menggunakan aplikasi digital setiap hari? "
if durasi_col not in df.columns:
    for col in df.columns:
        if "Berapa lama Anda menggunakan aplikasi digital setiap hari" in col:
            durasi_col = col
            break

df["durasi_jam"] = df[durasi_col].map(durasi_mapping)

# =====================================================
# FITUR MODEL
# =====================================================
feature_columns = [
    "sosial_media",
    "chatting",
    "streaming_video",
    "musik",
    "game_mobile",
    "ecommerce",
    "transportasi_online",
    "durasi_jam"
]
X = df[feature_columns]

# =====================================================
# SCALING & PREDIKSI
# =====================================================
X_scaled = scaler.transform(X)
clusters = model.predict(X_scaled)

df_result = df.copy()
df_result["Cluster"] = clusters

cluster_label = {
    0: "Pengguna Intensitas Tinggi",
    1: "Pengguna Intensitas Rendah",
    2: "Pengguna Intensitas Sedang"
}
df_result["Kategori"] = df_result["Cluster"].map(cluster_label)

sil_score = silhouette_score(X_scaled, clusters)

# =====================================================
# NAVBAR
# =====================================================
selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "Dataset",
        "Preprocessing",
        "Clustering",
        "Prediksi",
      
    ],
    icons=[
        "house",
        "table",
        "gear",
        "diagram-3",
        "person-plus",
        "graph-up"
    ],
    default_index=0,
    orientation="horizontal"
)

# =====================================================
# DASHBOARD
# =====================================================
if selected == "Dashboard":
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data", len(df_result))
    col2.metric("Jumlah Fitur", X.shape[1])
    col3.metric("Jumlah Cluster", len(np.unique(clusters)))
    col4.metric("Silhouette Score", round(sil_score, 4))
    
    # Penjelasan Silhouette Score di bawah metrik
    st.info(
        """
        **Silhouette Score** digunakan untuk mengukur kualitas clustering.
        Nilai mendekati 1 menunjukkan cluster yang terbentuk semakin baik 
        dan terpisah dengan jelas. Nilai yang diperoleh adalah **{}**, 
        yang berarti kualitas cluster cukup baik.
        """.format(round(sil_score, 4))
    )

    st.markdown("---")
    st.subheader("Distribusi Cluster")
    st.bar_chart(df_result["Kategori"].value_counts())

# =====================================================
# DATASET
# =====================================================
elif selected == "Dataset":
    st.header("Dataset")
    st.dataframe(df_result, use_container_width=True)

# =====================================================
# PREPROCESSING
# =====================================================
elif selected == "Preprocessing":
    st.header("Preprocessing")
    st.subheader("Data Setelah Transformasi")
    st.dataframe(X.head(), use_container_width=True)

    st.subheader("Data Setelah Scaling")
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    st.dataframe(scaled_df.head(), use_container_width=True)

# =====================================================
# CLUSTERING
# =====================================================
elif selected == "Clustering":
    st.header("Hasil Clustering")
    st.dataframe(df_result, use_container_width=True)

    st.subheader("Distribusi Kategori")
    st.bar_chart(df_result["Kategori"].value_counts())

    csv = df_result.to_csv(index=False)
    st.download_button(
        label="Download Hasil Clustering",
        data=csv,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )

# =====================================================
# PREDIKSI
# =====================================================
elif selected == "Prediksi":
    st.header("Prediksi Cluster untuk Responden Baru")
    st.write("Masukkan data responden baru sesuai dengan pola penggunaan aplikasi digital.")

    with st.form("prediksi_form"):
        col1, col2 = st.columns(2)

        with col1:
            sosial_media = st.radio(
                "Frekuensi Media Sosial (1=Sangat Jarang, 5=Sangat Sering)",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            chatting = st.radio(
                "Frekuensi Chatting",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            streaming_video = st.radio(
                "Frekuensi Streaming Video",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            musik = st.radio(
                "Frekuensi Musik",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )

        with col2:
            game_mobile = st.radio(
                "Frekuensi Game Mobile",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            ecommerce = st.radio(
                "Frekuensi E-commerce",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            transportasi_online = st.radio(
                "Frekuensi Transportasi Online",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True
            )
            durasi = st.selectbox(
                "Durasi Penggunaan per Hari",
                options=["< 1 jam", "1–3 jam", "3–5 jam", "5–7 jam", "> 7 jam"],
                index=2
            )

        # Aplikasi favorit tidak digunakan dalam model prediksi
        submitted = st.form_submit_button("Prediksi Cluster")

    if submitted:
        durasi_map = {"< 1 jam": 1, "1–3 jam": 2, "3–5 jam": 3, "5–7 jam": 4, "> 7 jam": 5}
        durasi_val = durasi_map[durasi]

        input_data = np.array([[
            sosial_media,
            chatting,
            streaming_video,
            musik,
            game_mobile,
            ecommerce,
            transportasi_online,
            durasi_val
        ]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        kategori = cluster_label[cluster]

        st.success(f"Hasil Prediksi: {kategori} (Cluster {cluster})")

        if cluster == 0:
            st.info("Pengguna Intensitas Tinggi – Sangat aktif, durasi panjang, frekuensi tinggi di berbagai aplikasi.")
        elif cluster == 1:
            st.info("Pengguna Intensitas Rendah – Penggunaan jarang, durasi pendek, cenderung hanya untuk kebutuhan dasar.")
        else:
            st.info("Pengguna Intensitas Sedang – Penggunaan cukup sering dengan durasi sedang dan variasi aplikasi.")

        # Simpan ke file CSV terpisah
        data_baru = pd.DataFrame([{
            "sosial_media": sosial_media,
            "chatting": chatting,
            "streaming_video": streaming_video,
            "musik": musik,
            "game_mobile": game_mobile,
            "ecommerce": ecommerce,
            "transportasi_online": transportasi_online,
            "durasi": durasi,
            "cluster": cluster,
            "kategori": kategori
        }])

        try:
            existing = pd.read_csv("prediksi_baru.csv")
            updated = pd.concat([existing, data_baru], ignore_index=True)
            updated.to_csv("prediksi_baru.csv", index=False)
        except FileNotFoundError:
            data_baru.to_csv("prediksi_baru.csv", index=False)

        st.success("Data prediksi berhasil disimpan ke prediksi_baru.csv")

    # Tampilkan riwayat prediksi
    st.subheader("Riwayat Prediksi Tersimpan")
    try:
        df_pred = pd.read_csv("prediksi_baru.csv")
        st.dataframe(df_pred, use_container_width=True)
    except FileNotFoundError:
        st.info("Belum ada data prediksi tersimpan.")

# =====================================================
# EVALUASI
# =====================================================
elif selected == "Evaluasi":
    st.header("Evaluasi Model")
    st.metric("Silhouette Score", round(sil_score, 4))
    st.info(
        """
        Silhouette Score digunakan untuk mengukur kualitas clustering.
        Nilai mendekati 1 menunjukkan cluster yang terbentuk semakin baik dan terpisah dengan jelas.
        """
    )