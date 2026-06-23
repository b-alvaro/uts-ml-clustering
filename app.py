import streamlit as st
import pandas as pd
import numpy as np
import joblib

from streamlit_option_menu import option_menu
from sklearn.metrics import silhouette_score

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    <h1 style='text-align:center'>
    📱 Clustering Penggunaan Aplikasi Digital Mahasiswa
    </h1>
    """,
    unsafe_allow_html=True
)

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
        "Evaluasi",

    ],
    icons=[
        "house",
        "table",
        "gear",
        "diagram-3",
        "graph-up",

    ],
    default_index=0,
    orientation="horizontal"
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📂 Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload dataset terlebih dahulu.")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(uploaded_file)

# Simpan data asli
df_result = df.copy()

# =====================================================
# RENAME KOLOM
# =====================================================

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

X = df[
    [
        "sosial_media",
        "chatting",
        "streaming_video",
        "musik",
        "game_mobile",
        "ecommerce",
        "transportasi_online",
        "durasi_jam"
    ]
]

# =====================================================
# SCALING
# =====================================================

X_scaled = scaler.transform(X)

# =====================================================
# PREDIKSI
# =====================================================

clusters = model.predict(X_scaled)

df_result["Cluster"] = clusters

cluster_label = {
    0: "Pengguna Intensitas Tinggi",
    1: "Pengguna Intensitas Rendah",
    2: "Pengguna Intensitas Sedang"
}

df_result["Kategori"] = df_result["Cluster"].map(cluster_label)

# =====================================================
# EVALUASI
# =====================================================

sil_score = silhouette_score(X_scaled, clusters)

# =====================================================
# DASHBOARD
# =====================================================

if selected == "Dashboard":

    st.header("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Data",
        len(df_result)
    )

    col2.metric(
        "Jumlah Fitur",
        X.shape[1]
    )

    col3.metric(
        "Jumlah Cluster",
        len(np.unique(clusters))
    )

    col4.metric(
        "Silhouette Score",
        round(sil_score, 4)
    )

    st.markdown("---")

    st.subheader("Distribusi Cluster")

    st.bar_chart(
        df_result["Kategori"].value_counts()
    )

# =====================================================
# DATASET
# =====================================================

elif selected == "Dataset":

    st.header("📁 Dataset")

    st.dataframe(
        df_result,
        use_container_width=True
    )

# =====================================================
# PREPROCESSING
# =====================================================

elif selected == "Preprocessing":

    st.header("🔧 Preprocessing")

    st.subheader("Data Setelah Transformasi")

    st.dataframe(
        X.head(),
        use_container_width=True
    )

    st.subheader("Data Setelah Scaling")

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    st.dataframe(
        scaled_df.head(),
        use_container_width=True
    )

# =====================================================
# CLUSTERING
# =====================================================

elif selected == "Clustering":

    st.header("🤖 Hasil Clustering")

    st.dataframe(
        df_result,
        use_container_width=True
    )

    st.subheader("Distribusi Kategori")

    st.bar_chart(
        df_result["Kategori"].value_counts()
    )

    csv = df_result.to_csv(index=False)

    st.download_button(
        label="📥 Download Hasil Clustering",
        data=csv,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )

# =====================================================
# EVALUASI
# =====================================================

elif selected == "Evaluasi":

    st.header("📈 Evaluasi Model")

    st.metric(
        "Silhouette Score",
        round(sil_score, 4)
    )

    st.info(
        """
        Silhouette Score digunakan untuk mengukur
        kualitas clustering.

        Nilai mendekati 1 menunjukkan cluster yang
        terbentuk semakin baik dan terpisah dengan jelas.
        """
    )

