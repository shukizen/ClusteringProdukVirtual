import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score, davies_bouldin_score

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Segmentasi RFM & K-Means",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI (Modern, Fresh, Glassmorphism)
st.markdown("""
<style>
    /* Import Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main Header with Gradient */
    .main-header {
        font-size: 36px !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #2563EB, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
    }
    
    .sub-header {
        font-size: 16px !important;
        font-weight: 500;
        color: #64748B;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 30px;
    }
    
    /* Metric Cards with Elegant Solid Design */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
        margin-bottom: 15px;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #2563EB; /* Solid Blue instead of gradient */
        margin-top: 5px;
        margin-bottom: 0px;
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* Style Streamlit Tabs for a cleaner look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        border-bottom: 3px solid transparent;
        color: #64748B;
        font-weight: 600;
        padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid #3B82F6 !important;
        color: #3B82F6 !important;
    }
    
    /* Native Metrics override for consistency (White Text for Evaluasi Model) */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CACHED FUNCTIONS (Data Processing)
# ==========================================

@st.cache_data
def load_and_preprocess_data(uploaded_files):
    """Membaca dan menggabungkan semua file Excel yang diunggah, lalu melakukan pembersihan data awal."""
    if not uploaded_files:
        return None
    
    dataframes = []
    for file in uploaded_files:
        try:
            df_temp = pd.read_excel(file)
            dataframes.append(df_temp)
        except Exception as e:
            st.error(f"Error membaca {file.name}: {e}")
            
    if not dataframes:
        return None
        
    df = pd.concat(dataframes, ignore_index=True)
    
    # Preprocessing
    if 'Status_Pesanan' in df.columns:
        df = df[df['Status_Pesanan'] == 'Pesanan selesai'].copy()
        
    if 'Tanggal_Dibuat' in df.columns:
        df['Tanggal_Dibuat'] = pd.to_datetime(df['Tanggal_Dibuat'])
        
    return df

@st.cache_data
def calculate_rfm(df):
    """Menghitung metrik RFM (Recency, Frequency, Monetary) dari dataframe transaksi."""
    # Menentukan snapshot date (1 hari setelah transaksi terakhir)
    snapshot_date = df['Tanggal_Dibuat'].max() + datetime.timedelta(days=1)
    
    # Agregasi RFM
    rfm = df.groupby('Nama_Pembeli').agg({
        'Tanggal_Dibuat': lambda x: (snapshot_date - x.max()).days,
        'Nomor_Pesanan': 'count',
        'Total_Pendapatan': 'sum'
    }).reset_index()
    
    # Rename kolom
    rfm.rename(columns={
        'Tanggal_Dibuat': 'Recency',
        'Nomor_Pesanan': 'Frequency',
        'Total_Pendapatan': 'Monetary'
    }, inplace=True)
    
    return rfm

@st.cache_data
def normalize_and_cluster(rfm, k_clusters):
    """Melakukan normalisasi (log + StandardScaler) dan K-Means clustering."""
    # Filter data dengan Monetary > 0 (menghindari error log1p jika ada nilai negatif, meski seharusnya tidak ada)
    rfm_clean = rfm[rfm['Monetary'] >= 0].copy()
    
    # Ekstraksi fitur
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_clean[features]
    
    # Log Transformation untuk menangani skewness
    X_log = np.log1p(X)
    
    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    rfm_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Mengurutkan cluster berdasarkan rata-rata Monetary (untuk konsistensi label)
    cluster_means = rfm_clean.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    # Mapping cluster lama ke cluster baru (0 tertinggi, k-1 terendah)
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(cluster_means.index)}
    rfm_clean['Cluster'] = rfm_clean['Cluster'].map(cluster_map)
    
    # Menentukan label segmen berdasarkan urutan cluster
    def assign_segment(cluster):
        if k_clusters == 2:
            labels = {
                0: 'High Value Customers (Core)',
                1: 'Low Value / Hibernating'
            }
            return labels.get(cluster, f'Segmen {cluster}')
        elif k_clusters == 3:
            labels = {
                0: 'Champions / High Value',
                1: 'Potential Loyalist / At Risk',
                2: 'Lost / Hibernating'
            }
            return labels.get(cluster, f'Segmen {cluster}')
        elif k_clusters == 4:
            labels = {0: 'Champions', 1: 'Loyal Customers', 2: 'At Risk', 3: 'Lost/Hibernating'}
            return labels.get(cluster, f'Segmen {cluster}')
        elif k_clusters == 5:
            labels = {0: 'Champions', 1: 'Loyal Customers', 2: 'Potential Loyalist', 3: 'At Risk', 4: 'Lost/Hibernating'}
            return labels.get(cluster, f'Segmen {cluster}')
        elif k_clusters == 6:
            labels = {
                0: 'Champions (VIP)',
                1: 'Loyal Customers',
                2: 'Potential Loyalist',
                3: 'New / Recent Customers',
                4: 'At Risk',
                5: 'Lost/Hibernating'
            }
            return labels.get(cluster, f'Segmen {cluster}')
        else:
            # Fallback untuk K > 6
            if cluster == 0:
                return 'Champions (VIP)'
            elif cluster == 1:
                return 'Loyal Customers'
            elif cluster == 2:
                return 'Potential Loyalist'
            elif cluster == k_clusters - 2:
                return 'At Risk'
            elif cluster == k_clusters - 1:
                return 'Lost/Hibernating'
            else:
                return f'Segmen Menengah {cluster}'
            
    rfm_clean['Segmen'] = rfm_clean['Cluster'].apply(assign_segment)
    
    return rfm_clean, X_scaled, kmeans

@st.cache_data
def calculate_optimal_k_metrics(X_scaled):
    """Menghitung Inertia dan Silhouette Score untuk K dari 2 sampai 8 untuk Analisis Elbow."""
    inertias = []
    silhouette_scores = []
    k_range = list(range(2, 9))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        
    return k_range, inertias, silhouette_scores

# ==========================================
# 2. SIDEBAR (Input & Konfigurasi)
# ==========================================

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("### 1. Unggah Data Transaksi")
    uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type=['xlsx'], accept_multiple_files=True)

# Load data first if files are uploaded
df = None
if uploaded_files:
    df = load_and_preprocess_data(uploaded_files)

# Conditionally display filters and model parameters in the sidebar
if df is not None and not df.empty:
    with st.sidebar:
        st.markdown("### 2. Pengaturan Model")
        k_clusters = st.slider("Jumlah Segmen (K)", min_value=2, max_value=8, value=4, step=1, 
                               help="Tentukan berapa banyak kelompok pelanggan yang ingin dibentuk.")
        
        st.markdown("---")
        st.info("💡 **Tip**: Pastikan file yang diunggah adalah hasil ekspor riwayat pesanan dari platform.")

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================

st.markdown('<p class="main-header">🛍️ Dashboard Segmentasi Pelanggan Toko Hatta</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis RFM dan K-Means Clustering untuk Strategi Penjualan Produk Virtual</p>', unsafe_allow_html=True)

if not uploaded_files:
    st.warning("⚠️ Silakan unggah satu atau lebih file Excel riwayat transaksi pada panel sebelah kiri untuk memulai analisis.")
    
    # Tampilkan ilustrasi atau panduan kosong
    st.markdown("""
    ### 👋 Selamat Datang di Aplikasi Analisis Pelanggan
    Aplikasi ini dirancang untuk membantu Anda memahami perilaku pelanggan berdasarkan data transaksi historis.
    
    **Langkah-langkah:**
    1. Ekspor data riwayat pesanan (format `.xlsx`).
    2. Unggah file tersebut melalui menu di sidebar kiri.
    3. Biarkan sistem memproses data secara otomatis (Preprocessing, RFM, K-Means).
    4. Jelajahi *insight* pada tab-tab yang disediakan.
    """)
else:
    with st.spinner("Memproses data..."):
        if df is None or df.empty:
            st.error("Data kosong atau terjadi kesalahan saat membaca file. Pastikan format file sesuai.")
            st.stop()
            
        # Hitung RFM
        rfm = calculate_rfm(df)
        
        # Clustering
        rfm_clustered, rfm_scaled, kmeans_model = normalize_and_cluster(rfm, k_clusters)
        
    # Buat Tab UI
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Data", "🎯 Analisis RFM", "🧩 Hasil Segmentasi", "💡 Rekomendasi Strategi"])
    
    # ------------------------------------------
    # TAB 1: RINGKASAN DATA (EDA)
    # ------------------------------------------
    with tab1:
        st.markdown("### Ringkasan Performa Keseluruhan")
        
        # Metrik Utama
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Pendapatan</div>
                <div class="metric-value">Rp {df['Total_Pendapatan'].sum():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Pesanan</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pelanggan Unik</div>
                <div class="metric-value">{df['Nama_Pembeli'].nunique():,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_order_value = df['Total_Pendapatan'].sum() / len(df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rata-rata Nilai Pesanan</div>
                <div class="metric-value">Rp {avg_order_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # Spacer
        
        # Visualisasi Tren Penjualan
        st.markdown("#### Tren Penjualan")
        df_trend = df.set_index('Tanggal_Dibuat').resample('M')['Total_Pendapatan'].sum().reset_index()
        df_trend['Bulan'] = df_trend['Tanggal_Dibuat'].dt.strftime('%b %Y')
        
        fig_trend = px.line(df_trend, x='Bulan', y='Total_Pendapatan', markers=True,
                            title="Tren Pendapatan Bulanan",
                            labels={'Total_Pendapatan': 'Total Pendapatan (Rp)'},
                            color_discrete_sequence=['#2563EB'])
        fig_trend.update_layout(xaxis_title="", yaxis_title="Pendapatan", hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Bar Charts untuk Produk & Pembeli Top
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 10 Produk Terlaris")
            top_products = df['Nama_Produk'].value_counts().head(10).reset_index()
            top_products.columns = ['Nama_Produk', 'Jumlah_Transaksi']
            fig_prod = px.bar(top_products, x='Jumlah_Transaksi', y='Nama_Produk', orientation='h',
                              color='Jumlah_Transaksi', color_continuous_scale='Viridis')
            fig_prod.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_prod, use_container_width=True)
            
        with col_b:
            st.markdown("#### 10 Pembeli Teraktif")
            top_buyers = df['Nama_Pembeli'].value_counts().head(10).reset_index()
            top_buyers.columns = ['Nama_Pembeli', 'Jumlah_Transaksi']
            top_buyers['Nama_Pembeli'] = top_buyers['Nama_Pembeli'].astype(str) # Handle integer names
            fig_buyer = px.bar(top_buyers, x='Jumlah_Transaksi', y='Nama_Pembeli', orientation='h',
                               color='Jumlah_Transaksi', color_continuous_scale='Magma')
            fig_buyer.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_buyer, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Eksplorasi Waktu Transaksi & Preferensi Layanan")
        col_c, col_d, col_e = st.columns(3)
        
        with col_c:
            st.markdown("##### Tren Transaksi per Hari")
            if 'Tanggal_Dibuat' in df.columns:
                df['Hari'] = df['Tanggal_Dibuat'].dt.day_name()
                hari_map = {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu', 
                            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'}
                df['Hari_ID'] = df['Hari'].map(hari_map)
                sorter = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                hari_counts = df['Hari_ID'].value_counts().reindex(sorter).reset_index()
                hari_counts.columns = ['Hari', 'Jumlah Transaksi']
                fig_hari = px.bar(hari_counts, x='Hari', y='Jumlah Transaksi', 
                                  color='Jumlah Transaksi', color_continuous_scale='Blues')
                fig_hari.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
                st.plotly_chart(fig_hari, use_container_width=True)
            
        with col_d:
            if 'Kategori' in df.columns:
                st.markdown("##### Proporsi Kategori Produk")
                kategori_counts = df['Kategori'].value_counts().reset_index()
                kategori_counts.columns = ['Kategori', 'Jumlah']
                fig_kat = px.pie(kategori_counts, names='Kategori', values='Jumlah', hole=0.5,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_kat.update_traces(textposition='inside', textinfo='percent+label')
                fig_kat.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_kat, use_container_width=True)
                
        with col_e:
            if 'Pembeli_Premium' in df.columns:
                st.markdown("##### Tipe Pembeli (Premium)")
                prem_counts = df['Pembeli_Premium'].value_counts().reset_index()
                prem_counts.columns = ['Tipe', 'Jumlah']
                fig_prem = px.pie(prem_counts, names='Tipe', values='Jumlah', hole=0.5,
                                   color_discrete_sequence=px.colors.qualitative.Set2)
                fig_prem.update_traces(textposition='inside', textinfo='percent+label')
                fig_prem.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_prem, use_container_width=True)

    # ------------------------------------------
    # TAB 2: ANALISIS RFM
    # ------------------------------------------
    with tab2:
        st.markdown("### Analisis RFM (Recency, Frequency, Monetary)")
        st.write("""
        Model RFM adalah teknik pemodelan perilaku pelanggan yang didasarkan pada tiga metrik:
        - **Recency (R)**: Berapa hari sejak pelanggan terakhir kali melakukan pembelian. (Lebih rendah lebih baik)
        - **Frequency (F)**: Seberapa sering pelanggan melakukan pembelian. (Lebih tinggi lebih baik)
        - **Monetary (M)**: Berapa banyak uang yang dihabiskan pelanggan. (Lebih tinggi lebih baik)
        """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Statistik RFM")
            st.dataframe(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2).T, use_container_width=True)
            
        with col2:
            st.markdown("#### Distribusi RFM")
            fig_dist = make_subplots(rows=1, cols=3, subplot_titles=('Recency (Hari)', 'Frequency (Transaksi)', 'Monetary (Rupiah)'))
            
            fig_dist.add_trace(go.Histogram(x=rfm['Recency'], nbinsx=30, name="Recency", marker_color='#3B82F6'), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=rfm['Frequency'], nbinsx=30, name="Frequency", marker_color='#10B981'), row=1, col=2)
            fig_dist.add_trace(go.Histogram(x=rfm['Monetary'], nbinsx=30, name="Monetary", marker_color='#F59E0B'), row=1, col=3)
            
            fig_dist.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        # Korelasi antar Metrik RFM
        st.markdown("#### Korelasi antar Metrik (Heatmap)")
        corr_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto='.4f', 
                             color_continuous_scale='RdBu_r', 
                             range_color=[-1, 1],
                             labels=dict(color="Koefisien Korelasi"))
        fig_corr.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Hubungan antar Metrik RFM (Scatter Plots)")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            fig_sc1 = px.scatter(rfm, x='Recency', y='Frequency', opacity=0.5, color_discrete_sequence=['#8B5CF6'])
            fig_sc1.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_sc1, use_container_width=True)
        with col_s2:
            fig_sc2 = px.scatter(rfm, x='Recency', y='Monetary', opacity=0.5, color_discrete_sequence=['#EC4899'])
            fig_sc2.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_sc2, use_container_width=True)
        with col_s3:
            fig_sc3 = px.scatter(rfm, x='Frequency', y='Monetary', opacity=0.5, color_discrete_sequence=['#14B8A6'])
            fig_sc3.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_sc3, use_container_width=True)

    # ------------------------------------------
    # TAB 3: HASIL SEGMENTASI
    # ------------------------------------------
    with tab3:
        st.markdown(f"### Hasil K-Means Clustering (K={k_clusters})")
        
        with st.expander("🔍 Cari Jumlah Cluster Optimal (Metode Elbow & Silhouette)", expanded=False):
            st.write("Gunakan visualisasi ini untuk membantu menentukan jumlah kelompok (K) terbaik secara ilmiah:")
            k_range, inertias, sil_scores = calculate_optimal_k_metrics(rfm_scaled)
            
            # Elbow Chart
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers', 
                                           line=dict(color='#3B82F6', width=3),
                                           marker=dict(size=8)))
            fig_elbow.update_layout(title="Metode Elbow (WCSS / Inertia)",
                                    xaxis_title="Jumlah Cluster (K)",
                                    yaxis_title="Inertia (WCSS)",
                                    height=300, margin=dict(l=20, r=20, t=40, b=20))
            
            # Silhouette Chart
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=k_range, y=sil_scores, mode='lines+markers', 
                                         line=dict(color='#10B981', width=3),
                                         marker=dict(size=8)))
            fig_sil.update_layout(title="Silhouette Score per Cluster (K)",
                                  xaxis_title="Jumlah Cluster (K)",
                                  yaxis_title="Silhouette Score",
                                  height=300, margin=dict(l=20, r=20, t=40, b=20))
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.plotly_chart(fig_elbow, use_container_width=True)
            with col_e2:
                st.plotly_chart(fig_sil, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Proporsi Segmen Pelanggan")
            seg_counts = rfm_clustered['Segmen'].value_counts().reset_index()
            seg_counts.columns = ['Segmen', 'Jumlah Pelanggan']
            fig_pie = px.pie(seg_counts, values='Jumlah Pelanggan', names='Segmen', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("#### Karakteristik Rata-rata per Segmen")
            seg_mean = rfm_clustered.groupby('Segmen')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()
            st.dataframe(seg_mean, use_container_width=True, hide_index=True)
            
            # Heatmap Rata-rata RFM per Segmen (Skala Relatif)
            st.markdown("##### Visualisasi Heatmap (Skala Relatif)")
            seg_mean_scaled = seg_mean.set_index('Segmen').copy()
            for col in ['Recency', 'Frequency', 'Monetary']:
                min_val = seg_mean_scaled[col].min()
                max_val = seg_mean_scaled[col].max()
                seg_mean_scaled[col] = (seg_mean_scaled[col] - min_val) / (max_val - min_val + 1e-10)
                
            fig_heat = px.imshow(seg_mean_scaled.T,
                                 color_continuous_scale='YlGnBu',
                                 y=['Recency', 'Frequency', 'Monetary'],
                                 x=seg_mean_scaled.index.tolist(),
                                 labels=dict(color="Nilai Relatif"))
            
            actual_vals = seg_mean.set_index('Segmen').T.values
            fig_heat.update_traces(
                customdata=actual_vals,
                hovertemplate="Segmen: %{x}<br>Metrik: %{y}<br>Nilai Relatif: %{z:.2f}<br>Nilai Aktual: %{customdata:,.2f}<extra></extra>"
            )
            fig_heat.update_layout(height=350, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_heat, use_container_width=True)
            
        # Radar Chart
        st.markdown("#### Profil Segmen (Radar Chart)")
        
        # Normalisasi Min-Max khusus untuk radar chart agar skalanya seragam (0-1)
        radar_df = seg_mean.copy()
        for col in ['Recency', 'Frequency', 'Monetary']:
            # Khusus Recency, nilai lebih kecil = lebih baik, maka kita balik
            if col == 'Recency':
                radar_df[col] = 1 - ((radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min() + 1e-10))
            else:
                radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min() + 1e-10)
                
        fig_radar = go.Figure()
        categories = ['Recency (Inverted)', 'Frequency', 'Monetary']
        
        for i, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Recency'], row['Frequency'], row['Monetary']],
                theta=categories,
                fill='toself',
                name=row['Segmen']
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 3D PCA Scatter Plot
        st.markdown("#### Visualisasi Persebaran Cluster (PCA 3D)")
        st.write("Visualisasi ini menggunakan Principal Component Analysis (PCA) untuk mereduksi 3 dimensi RFM yang telah dinormalisasi menjadi komponen spasial agar dapat dilihat persebarannya.")
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(rfm_scaled)
        
        rfm_clustered['PCA1'] = pca_result[:, 0]
        rfm_clustered['PCA2'] = pca_result[:, 1]
        rfm_clustered['PCA3'] = pca_result[:, 2]
        
        fig_3d = px.scatter_3d(
            rfm_clustered, x='PCA1', y='PCA2', z='PCA3',
            color='Segmen', opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hover_data=['Recency', 'Frequency', 'Monetary']
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Evaluasi Model Clustering (Sama seperti di notebook)
        st.markdown("#### 📈 Evaluasi Model Clustering (Reconstruction Metrics)")
        st.write("Metrik evaluasi untuk mengukur seberapa presisi pembagian kluster menggunakan pendekatan rekonstruksi spasial:")
        
        sil_score_val = silhouette_score(rfm_scaled, rfm_clustered['Cluster'])
        db_score_val = davies_bouldin_score(rfm_scaled, rfm_clustered['Cluster'])
        
        centroids_val = kmeans_model.cluster_centers_
        rfm_scaled_pred = centroids_val[rfm_clustered['Cluster'].values]
        
        mae_val = mean_absolute_error(rfm_scaled, rfm_scaled_pred)
        mse_val = mean_squared_error(rfm_scaled, rfm_scaled_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(rfm_scaled, rfm_scaled_pred, multioutput='uniform_average')
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Silhouette Score", f"{sil_score_val:.4f}", help="Kekuatan struktur cluster (lebih mendekati 1 lebih baik)")
            st.metric("Davies-Bouldin Index", f"{db_score_val:.4f}", help="Rasio jarak dalam vs antar cluster (lebih kecil lebih baik)")
        with col_m2:
            st.metric("Mean Absolute Error (MAE)", f"{mae_val:.4f}", help="Rata-rata kesalahan absolut rekonstruksi data terstandarisasi")
            st.metric("Mean Squared Error (MSE)", f"{mse_val:.4f}", help="Rata-rata kesalahan kuadrat rekonstruksi data terstandarisasi")
        with col_m3:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse_val:.4f}", help="Akar rata-rata kesalahan kuadrat rekonstruksi")
            st.metric("R2 Score (Variance Explained)", f"{r2_val:.4f}", help="Proporsi varians data terstandarisasi yang dijelaskan oleh kluster (0-1)")

    # ------------------------------------------
    # TAB 4: REKOMENDASI & EKSPOR
    # ------------------------------------------
    with tab4:
        st.markdown("### Strategi Bisnis Berdasarkan Segmen")
        
        # Define strategies dynamically based on cluster label
        strategies = {
            "Champions": "Berikan reward khusus (program VIP, early access produk baru). Jangan beri terlalu banyak diskon harga, fokus pada eksklusivitas dan layanan premium.",
            "Champions (VIP)": "Berikan reward khusus (program VIP, early access produk baru). Jangan beri terlalu banyak diskon harga, fokus pada eksklusivitas dan layanan premium.",
            "High Value Customers (Core)": "Berikan reward loyalitas, pertahankan interaksi secara berkala, dan tawarkan paket bundling bernilai tinggi untuk memaksimalkan retensi.",
            "Champions / High Value": "Pertahankan kepuasan mereka dengan memberikan pelayanan prioritas, program VIP, dan penawaran eksklusif tanpa perang harga.",
            "Loyal Customers": "Tawarkan program loyalitas, upsell produk dengan margin lebih tinggi, dan minta ulasan atau testimoni positif.",
            "Potential Loyalist": "Berikan penawaran bundel (cross-selling) dan rekomendasikan produk lain untuk meningkatkan keranjang belanja (Monetary).",
            "Potential Loyalist / At Risk": "Tawarkan promosi berkala yang menarik minat mereka kembali, atau buat bundling hemat untuk memicu transaksi.",
            "New / Recent Customers": "Kirimkan panduan sambutan, voucher pembelian kedua, dan survey kepuasan singkat untuk membangun loyalitas sejak awal.",
            "At Risk": "Kirim penawaran personalisasi, diskon re-engagement, atau email promosi untuk menarik mereka kembali berbelanja.",
            "Lost/Hibernating": "Fokus pengeluaran marketing minimal (jangan buang budget besar). Lakukan kampanye standar atau abaikan jika Customer Acquisition Cost (CAC) baru lebih murah.",
            "Low Value / Hibernating": "Gunakan promosi massal berbiaya rendah (seperti broadcast berkala). Jangan alokasikan budget promosi khusus/besar untuk kelompok ini.",
            "Lost / Hibernating": "Fokus pengeluaran marketing minimal. Hubungi kembali hanya melalui kampanye otomatis berbiaya sangat rendah."
        }
        
        # Create expandable sections for each segment found
        unique_segments = rfm_clustered['Segmen'].unique()
        
        for seg in sorted(unique_segments):
            with st.expander(f"📌 {seg}", expanded=True):
                # Cari karakteristik rata-rata
                stats = seg_mean[seg_mean['Segmen'] == seg].iloc[0]
                st.write(f"**Karakteristik Rata-rata:** Transaksi terakhir **{stats['Recency']:.0f} hari lalu**, berbelanja **{stats['Frequency']:.1f} kali**, dengan total **Rp {stats['Monetary']:,.0f}**.")
                
                # Coba dapatkan strategi dari dictionary, jika tidak ada (custom K), berikan template generik
                strat = strategies.get(seg, "Berikan promosi produk terkait yang relevan dan diskon bersyarat (misal minimal belanja) untuk meningkatkan loyalitas mereka.")
                st.success(f"**Tindakan Strategis:** {strat}")
                
        st.markdown("---")
        st.markdown("### 📥 Ekspor Data Hasil Segmentasi")
        st.write("Anda dapat mengunduh seluruh data pelanggan lengkap dengan label segmennya untuk kebutuhan operasional marketing lebih lanjut.")
        
        # Siapkan data untuk diunduh (buang kolom PCA)
        export_df = rfm_clustered.drop(columns=['PCA1', 'PCA2', 'PCA3'], errors='ignore')
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Unduh File CSV",
            data=csv,
            file_name="Hasil_Segmentasi_RFM.csv",
            mime="text/csv",
        )
        
        st.dataframe(export_df.head(10), use_container_width=True)
