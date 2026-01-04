"""
Dashboard Interactivo: Análisis Multinivel de Competidores con ABSA
====================================================================
Dashboard Streamlit con enfoque híbrido:
- Market Commonality (MC) a nivel de VINO
- Resource Similarity (RS) a nivel de BODEGA
- ABSA para extracción de aspectos de notas de cata

Para ejecutar:
    streamlit run app_multilevel.py

Autor: ACEDE BODEGAS 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuración SSL para NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="ENOLYTICS Multinivel - Análisis ABSA",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MÓDULO ABSA (Simplificado para Streamlit)
# =============================================================================

class WineABSAExtractor:
    """Extractor ABSA simplificado para el dashboard."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

        self.aspect_taxonomy = {
            'aroma': {'terms': ['aroma', 'aromas', 'nose', 'bouquet', 'fragrant', 'scent'],
                      'dimension': 'MC'},
            'sabor': {'terms': ['taste', 'flavor', 'palate', 'delicious', 'savory'],
                      'dimension': 'MC'},
            'final': {'terms': ['finish', 'aftertaste', 'length', 'persistent', 'lingering'],
                      'dimension': 'MC'},
            'valor': {'terms': ['value', 'price', 'worth', 'bargain', 'affordable'],
                      'dimension': 'MC'},
            'estructura': {'terms': ['structure', 'body', 'full-bodied', 'weight', 'dense'],
                          'dimension': 'RS'},
            'taninos': {'terms': ['tannin', 'tannins', 'tannic', 'grip', 'velvety', 'silky'],
                       'dimension': 'RS'},
            'acidez': {'terms': ['acid', 'acidity', 'fresh', 'crisp', 'bright', 'vibrant'],
                      'dimension': 'RS'},
            'crianza': {'terms': ['oak', 'barrel', 'aged', 'toast', 'vanilla', 'cedar'],
                       'dimension': 'RS'},
        }

        self._compile_patterns()

    def _compile_patterns(self):
        self.aspect_patterns = {}
        for aspect, config in self.aspect_taxonomy.items():
            pattern = r'\b(' + '|'.join(re.escape(term) for term in config['terms']) + r')\b'
            self.aspect_patterns[aspect] = re.compile(pattern, re.IGNORECASE)

    def extract_absa(self, text):
        if not text or not isinstance(text, str):
            return {}

        results = {}
        for aspect, pattern in self.aspect_patterns.items():
            matches = list(pattern.finditer(text))
            if matches:
                # Analizar sentimiento del contexto
                sentiments = []
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    scores = self.sia.polarity_scores(context)
                    sentiments.append(scores['compound'])

                results[aspect] = {
                    'sentiment': np.mean(sentiments),
                    'n_mentions': len(matches),
                    'dimension': self.aspect_taxonomy[aspect]['dimension']
                }

        return results

    def get_mc_aspects(self):
        return [asp for asp, conf in self.aspect_taxonomy.items() if conf['dimension'] == 'MC']

    def get_rs_aspects(self):
        return [asp for asp, conf in self.aspect_taxonomy.items() if conf['dimension'] == 'RS']


# =============================================================================
# MODELOS DE DEEP LEARNING
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=None):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim),
                                   nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim),
                                   nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


class DeepSoftClustering(nn.Module):
    def __init__(self, input_dim, n_clusters, latent_dim=10, alpha=1.0):
        super(DeepSoftClustering, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder = Autoencoder(input_dim, latent_dim)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        x_recon, z = self.autoencoder(x)
        q = self.soft_assignment(z)
        return x_recon, z, q

    def soft_assignment(self, z):
        z_exp = z.unsqueeze(1)
        c_exp = self.cluster_centers.unsqueeze(0)
        dist_sq = torch.sum((z_exp - c_exp) ** 2, dim=2)
        q = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)

    def encode(self, x):
        return self.autoencoder.encode(x)


def train_model(model, data_loader, n_epochs=50, lr=0.001, lambda_kl=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.train()

    for epoch in range(n_epochs):
        for batch_x, in data_loader:
            batch_x = batch_x.to(device)
            x_recon, z, q = model(batch_x)
            recon_loss = mse_loss(x_recon, batch_x)

            weight = q ** 2 / q.sum(dim=0, keepdim=True)
            p = (weight / weight.sum(dim=1, keepdim=True)).detach()
            kl_loss = torch.sum(p * torch.log(p / (q + 1e-10))) / batch_x.size(0)

            loss = recon_loss + lambda_kl * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================

@st.cache_data
def get_available_countries(data_path):
    df = pd.read_csv(data_path, usecols=['country'])
    countries = df['country'].dropna().unique().tolist()
    return ['Todos'] + sorted(countries)


@st.cache_data
def load_wines_data(data_path, country_filter='Spain', min_wines=3):
    """Carga datos a nivel de vino."""
    df = pd.read_csv(data_path)

    if country_filter != 'Todos':
        df = df[df['country'] == country_filter].copy()

    # Eliminar duplicados
    df = df.drop_duplicates(subset=['designation', 'winery'], keep='first')
    df['price'] = df['price'].fillna(df['price'].median())

    # Filtrar bodegas con mínimo de vinos
    winery_counts = df['winery'].value_counts()
    valid_wineries = winery_counts[winery_counts >= min_wines].index
    df = df[df['winery'].isin(valid_wineries)].copy()

    return df.reset_index(drop=True)


@st.cache_data
def extract_absa_features(_df, _absa_extractor):
    """Extrae features ABSA de las notas de cata."""
    df = _df.copy()

    # Inicializar columnas
    for aspect in _absa_extractor.aspect_taxonomy.keys():
        df[f'sent_{aspect}'] = np.nan

    # Extraer ABSA
    for idx, row in df.iterrows():
        text = row.get('description', '')
        if pd.notna(text):
            absa = _absa_extractor.extract_absa(text)
            for aspect, data in absa.items():
                df.loc[idx, f'sent_{aspect}'] = data['sentiment']

    return df


@st.cache_data
def prepare_winery_data(_df_wines, _absa_extractor):
    """Agrega datos a nivel bodega para RS."""
    rs_aspects = _absa_extractor.get_rs_aspects()

    agg_dict = {
        'points': ['mean', 'std', 'count'],
        'price': ['mean', 'std', 'min', 'max'],
        'variety': lambda x: len(x.dropna().unique()),
    }

    # Agregar aspectos RS
    for asp in rs_aspects:
        col = f'sent_{asp}'
        if col in _df_wines.columns:
            agg_dict[col] = 'mean'

    winery_agg = _df_wines.groupby('winery').agg(agg_dict).reset_index()

    # Aplanar columnas
    new_cols = ['winery']
    for col in winery_agg.columns[1:]:
        if isinstance(col, tuple):
            if col[1] == '<lambda>':
                new_cols.append('n_varieties')
            else:
                new_cols.append(f'{col[0]}_{col[1]}')
        else:
            new_cols.append(col)
    winery_agg.columns = new_cols

    # Calcular rango de precios
    if 'price_max' in winery_agg.columns and 'price_min' in winery_agg.columns:
        winery_agg['price_range'] = winery_agg['price_max'] - winery_agg['price_min']

    # Provincia principal
    def get_main_province(winery):
        wines = _df_wines[_df_wines['winery'] == winery]
        if len(wines) > 0:
            mode = wines['province'].mode()
            if len(mode) > 0:
                return mode.iloc[0]
        return 'Unknown'

    winery_agg['main_province'] = winery_agg['winery'].apply(get_main_province)

    # Variedad principal
    def get_main_variety(winery):
        wines = _df_wines[_df_wines['winery'] == winery]
        if len(wines) > 0:
            mode = wines['variety'].mode()
            if len(mode) > 0:
                return mode.iloc[0]
        return 'Unknown'

    winery_agg['main_variety'] = winery_agg['winery'].apply(get_main_variety)

    return winery_agg


def categorize_wine_type(variety):
    if pd.isna(variety):
        return 'other'
    v = str(variety).lower()
    if any(w in v for w in ['cabernet', 'merlot', 'tempranillo', 'syrah', 'pinot noir', 'garnacha']):
        return 'red'
    elif any(w in v for w in ['chardonnay', 'sauvignon', 'verdejo', 'albariño', 'riesling']):
        return 'white'
    elif 'rosé' in v or 'rosado' in v:
        return 'rose'
    elif any(w in v for w in ['cava', 'champagne', 'sparkling']):
        return 'sparkling'
    elif any(w in v for w in ['sherry', 'jerez', 'pedro ximénez', 'palomino']):
        return 'fortified'
    return 'other'


def run_multilevel_clustering(df_wines, df_wineries, absa_extractor,
                               n_clusters_mc=8, n_clusters_rs=6):
    """Ejecuta clustering dual MC (vinos) y RS (bodegas)."""

    # === MC FEATURES (nivel vino) ===
    mc_aspects = absa_extractor.get_mc_aspects()
    mc_features = ['points', 'price'] + [f'sent_{asp}' for asp in mc_aspects]

    df_wines['wine_type'] = df_wines['variety'].apply(categorize_wine_type)
    le_type = LabelEncoder()
    df_wines['wine_type_enc'] = le_type.fit_transform(df_wines['wine_type'])
    mc_features.append('wine_type_enc')

    X_mc = df_wines[mc_features].copy()
    for col in X_mc.columns:
        X_mc[col] = X_mc[col].fillna(X_mc[col].median())

    scaler_mc = StandardScaler()
    X_mc_scaled = scaler_mc.fit_transform(X_mc)

    # Clustering MC
    X_mc_tensor = torch.FloatTensor(X_mc_scaled)
    dataset_mc = TensorDataset(X_mc_tensor)
    loader_mc = DataLoader(dataset_mc, batch_size=64, shuffle=True)

    model_mc = DeepSoftClustering(X_mc_scaled.shape[1], n_clusters_mc, latent_dim=6).to(device)

    # Inicializar centroides
    with torch.no_grad():
        z = model_mc.encode(X_mc_tensor.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters_mc, random_state=42, n_init=10)
    kmeans.fit(z)
    model_mc.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    model_mc = train_model(model_mc, loader_mc, n_epochs=50)

    model_mc.eval()
    with torch.no_grad():
        _, _, q_mc = model_mc(X_mc_tensor.to(device))
        mc_assignments = q_mc.cpu().numpy()

    df_wines['mc_cluster'] = np.argmax(mc_assignments, axis=1)
    for i in range(n_clusters_mc):
        df_wines[f'mc_prob_{i}'] = mc_assignments[:, i]

    # === RS FEATURES (nivel bodega) ===
    rs_aspects = absa_extractor.get_rs_aspects()
    rs_features = ['n_varieties', 'points_count']
    if 'price_range' in df_wineries.columns:
        rs_features.append('price_range')

    for asp in rs_aspects:
        col = f'sent_{asp}_mean'
        if col in df_wineries.columns:
            rs_features.append(col)

    # Codificar variedad
    le_var = LabelEncoder()
    df_wineries['variety_enc'] = le_var.fit_transform(df_wineries['main_variety'])
    rs_features.append('variety_enc')

    X_rs = df_wineries[rs_features].copy()
    for col in X_rs.columns:
        X_rs[col] = X_rs[col].fillna(X_rs[col].median())

    scaler_rs = StandardScaler()
    X_rs_scaled = scaler_rs.fit_transform(X_rs)

    # Clustering RS
    X_rs_tensor = torch.FloatTensor(X_rs_scaled)
    dataset_rs = TensorDataset(X_rs_tensor)
    loader_rs = DataLoader(dataset_rs, batch_size=min(32, len(X_rs_scaled)), shuffle=True)

    model_rs = DeepSoftClustering(X_rs_scaled.shape[1], n_clusters_rs, latent_dim=4).to(device)

    with torch.no_grad():
        z = model_rs.encode(X_rs_tensor.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters_rs, random_state=42, n_init=10)
    kmeans.fit(z)
    model_rs.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    model_rs = train_model(model_rs, loader_rs, n_epochs=50)

    model_rs.eval()
    with torch.no_grad():
        _, _, q_rs = model_rs(X_rs_tensor.to(device))
        rs_assignments = q_rs.cpu().numpy()

    df_wineries['rs_cluster'] = np.argmax(rs_assignments, axis=1)
    for i in range(n_clusters_rs):
        df_wineries[f'rs_prob_{i}'] = rs_assignments[:, i]

    return df_wines, df_wineries, mc_assignments, rs_assignments, n_clusters_mc, n_clusters_rs


def analyze_focal_winery(target_winery, df_wines, df_wineries,
                          mc_assignments, rs_assignments):
    """Analiza competidores de una bodega focal con enfoque multinivel."""

    focal_idx = df_wineries[df_wineries['winery'] == target_winery].index[0]
    focal_wines = df_wines[df_wines['winery'] == target_winery]
    focal_wine_indices = focal_wines.index.tolist()
    focal_rs_probs = rs_assignments[focal_idx]

    # Calcular umbrales dinámicos
    # MC: mediana de similitudes entre vinos aleatorios
    sample_size = min(500, len(df_wines))
    sample_idx = np.random.choice(len(df_wines), sample_size, replace=False)
    mc_sims = []
    for i in range(0, len(sample_idx)-1, 2):
        sim = cosine_similarity([mc_assignments[sample_idx[i]]],
                               [mc_assignments[sample_idx[i+1]]])[0][0]
        mc_sims.append(sim)
    mc_threshold = np.median(mc_sims)

    # RS: mediana de similitudes entre bodegas
    rs_sims = []
    for i in range(len(rs_assignments)):
        for j in range(i+1, min(i+20, len(rs_assignments))):
            sim = cosine_similarity([rs_assignments[i]], [rs_assignments[j]])[0][0]
            rs_sims.append(sim)
    rs_threshold = np.median(rs_sims)

    # Analizar cada competidor
    results = []

    for comp_idx, comp_row in df_wineries.iterrows():
        if comp_row['winery'] == target_winery:
            continue

        comp_wines = df_wines[df_wines['winery'] == comp_row['winery']]
        comp_wine_indices = comp_wines.index.tolist()

        if len(comp_wine_indices) == 0:
            continue

        # RS entre bodegas
        rs_score = cosine_similarity([focal_rs_probs], [rs_assignments[comp_idx]])[0][0]

        # Calcular MC para pares de vinos y categorizar
        categories = {'Core': 0, 'Substitute': 0, 'Marginal': 0, 'Potential': 0}
        mc_scores = []

        for f_wine_idx in focal_wine_indices:
            for c_wine_idx in comp_wine_indices:
                mc_score = cosine_similarity([mc_assignments[f_wine_idx]],
                                            [mc_assignments[c_wine_idx]])[0][0]
                mc_scores.append(mc_score)

                mc_high = mc_score >= mc_threshold
                rs_high = rs_score >= rs_threshold

                if mc_high and rs_high:
                    categories['Core'] += 1
                elif mc_high and not rs_high:
                    categories['Substitute'] += 1
                elif not mc_high and rs_high:
                    categories['Marginal'] += 1
                else:
                    categories['Potential'] += 1

        n_pairs = sum(categories.values())
        if n_pairs == 0:
            continue

        profile = {cat: count / n_pairs for cat, count in categories.items()}
        modal_category = max(categories, key=categories.get)

        # Dispersión
        probs = np.array(list(profile.values()))
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        dispersion = entropy / np.log(4) if np.log(4) > 0 else 0

        # Intensidad
        intensity = (1.0 * profile['Core'] + 0.7 * profile['Substitute'] +
                    0.4 * profile['Marginal'] + 0.1 * profile['Potential'])

        results.append({
            'competitor': comp_row['winery'],
            'n_wines': len(comp_wine_indices),
            'rs_score': rs_score,
            'avg_mc_score': np.mean(mc_scores),
            'pct_core': profile['Core'],
            'pct_substitute': profile['Substitute'],
            'pct_marginal': profile['Marginal'],
            'pct_potential': profile['Potential'],
            'modal_category': modal_category,
            'dispersion': dispersion,
            'intensity': intensity,
            'price_mean': comp_row.get('price_mean', 0),
            'points_mean': comp_row.get('points_mean', 0),
            'main_province': comp_row.get('main_province', 'Unknown'),
            'main_variety': comp_row.get('main_variety', 'Unknown'),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('intensity', ascending=False)

    return df_results, mc_threshold, rs_threshold


# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

def main():
    # Header
    st.title("🍷 ENOLYTICS - Análisis Multinivel de Competidores")
    st.markdown("""
    **Enfoque Híbrido ABSA + Kamensky**
    - Market Commonality (MC) a nivel de **VINO**
    - Resource Similarity (RS) a nivel de **BODEGA**
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuración")

    # Data path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "winemag-data_first150k.csv")

    if not os.path.exists(data_path):
        # Intentar ruta alternativa
        data_path = "/Users/antoniorafaelramosrodriguez/Dropbox/ACEDE BODEGAS 2026/DATA/winemag-data_first150k.csv"

    if not os.path.exists(data_path):
        st.error("No se encontró el archivo de datos.")
        st.stop()

    # Filtros
    countries = get_available_countries(data_path)
    country_filter = st.sidebar.selectbox("País", countries, index=countries.index('Spain') if 'Spain' in countries else 0)

    min_wines = st.sidebar.slider("Mínimo vinos por bodega", 2, 10, 3)

    n_clusters_mc = st.sidebar.slider("Clusters MC (vinos)", 4, 15, 8)
    n_clusters_rs = st.sidebar.slider("Clusters RS (bodegas)", 3, 10, 6)

    # Cargar datos
    with st.spinner("Cargando datos..."):
        df_wines = load_wines_data(data_path, country_filter, min_wines)

    st.sidebar.markdown(f"**Vinos cargados:** {len(df_wines):,}")
    st.sidebar.markdown(f"**Bodegas:** {df_wines['winery'].nunique():,}")

    # Selector de bodega
    wineries_list = sorted(df_wines['winery'].unique().tolist())
    default_winery = 'Hidalgo' if 'Hidalgo' in wineries_list else wineries_list[0]
    target_winery = st.sidebar.selectbox("Bodega Focal", wineries_list,
                                         index=wineries_list.index(default_winery))

    # Botón para ejecutar análisis
    if st.sidebar.button("🚀 Ejecutar Análisis Multinivel", type="primary"):
        # ABSA Extractor
        absa_extractor = WineABSAExtractor()

        # Extraer ABSA
        with st.spinner("Extrayendo aspectos de notas de cata (ABSA)..."):
            df_wines_absa = extract_absa_features(df_wines, absa_extractor)

        # Preparar datos de bodegas
        with st.spinner("Preparando datos de bodegas..."):
            df_wineries = prepare_winery_data(df_wines_absa, absa_extractor)

        # Clustering dual
        with st.spinner("Ejecutando Deep Soft Clustering dual..."):
            (df_wines_clust, df_wineries_clust,
             mc_assignments, rs_assignments,
             n_mc, n_rs) = run_multilevel_clustering(
                df_wines_absa, df_wineries, absa_extractor,
                n_clusters_mc, n_clusters_rs
            )

        # Análisis focal
        with st.spinner(f"Analizando competidores de {target_winery}..."):
            df_profiles, mc_thresh, rs_thresh = analyze_focal_winery(
                target_winery, df_wines_clust, df_wineries_clust,
                mc_assignments, rs_assignments
            )

        # Guardar en session state
        st.session_state['df_wines'] = df_wines_clust
        st.session_state['df_wineries'] = df_wineries_clust
        st.session_state['df_profiles'] = df_profiles
        st.session_state['target_winery'] = target_winery
        st.session_state['mc_thresh'] = mc_thresh
        st.session_state['rs_thresh'] = rs_thresh
        st.session_state['mc_assignments'] = mc_assignments
        st.session_state['rs_assignments'] = rs_assignments

    # Mostrar resultados si existen
    if 'df_profiles' in st.session_state:
        df_profiles = st.session_state['df_profiles']
        target_winery = st.session_state['target_winery']
        df_wines = st.session_state['df_wines']
        df_wineries = st.session_state['df_wineries']

        # === TABS DE RESULTADOS ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen", "🎯 Competidores", "🍇 Vinos", "📈 Visualizaciones"
        ])

        with tab1:
            st.header(f"Análisis Multinivel: {target_winery}")

            # Perfil de la bodega focal
            focal_data = df_wineries[df_wineries['winery'] == target_winery].iloc[0]
            focal_wines = df_wines[df_wines['winery'] == target_winery]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vinos", len(focal_wines))
            col2.metric("Precio Medio", f"${focal_data.get('price_mean', 0):.0f}")
            col3.metric("Puntuación Media", f"{focal_data.get('points_mean', 0):.1f}")
            col4.metric("Cluster RS", int(focal_data.get('rs_cluster', 0)) + 1)

            # Distribución de clusters MC de vinos
            st.subheader("Distribución de vinos por cluster MC")
            mc_dist = focal_wines['mc_cluster'].value_counts().sort_index()
            fig_mc = px.bar(x=[f"MC_{i+1}" for i in mc_dist.index],
                           y=mc_dist.values,
                           labels={'x': 'Cluster MC', 'y': 'Nº Vinos'},
                           color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_mc, use_container_width=True)

            # Resumen de competidores
            st.subheader("Resumen de Competidores")
            col1, col2, col3, col4 = st.columns(4)

            n_core = len(df_profiles[df_profiles['modal_category'] == 'Core'])
            n_sub = len(df_profiles[df_profiles['modal_category'] == 'Substitute'])
            n_marg = len(df_profiles[df_profiles['modal_category'] == 'Marginal'])
            n_pot = len(df_profiles[df_profiles['modal_category'] == 'Potential'])

            col1.metric("Core", n_core, delta=None)
            col2.metric("Substitute", n_sub, delta=None)
            col3.metric("Marginal", n_marg, delta=None)
            col4.metric("Potential", n_pot, delta=None)

        with tab2:
            st.header("Perfiles de Competidores")

            # Filtro por categoría
            cat_filter = st.selectbox("Filtrar por categoría modal",
                                      ['Todos', 'Core', 'Substitute', 'Marginal', 'Potential'])

            df_show = df_profiles.copy()
            if cat_filter != 'Todos':
                df_show = df_show[df_show['modal_category'] == cat_filter]

            # Tabla interactiva
            st.dataframe(
                df_show[['competitor', 'modal_category', 'intensity', 'dispersion',
                        'pct_core', 'pct_substitute', 'pct_marginal', 'pct_potential',
                        'rs_score', 'avg_mc_score', 'price_mean', 'main_variety']].round(3),
                use_container_width=True,
                height=400
            )

            # Top 5 por categoría
            st.subheader("Top 5 por Categoría")

            for cat in ['Core', 'Substitute', 'Marginal', 'Potential']:
                with st.expander(f"🏆 Top 5 {cat}"):
                    top5 = df_profiles[df_profiles['modal_category'] == cat].head(5)
                    for i, (_, row) in enumerate(top5.iterrows(), 1):
                        st.markdown(f"""
                        **{i}. {row['competitor']}**
                        - Intensidad: {row['intensity']:.3f} | Dispersión: {row['dispersion']:.2f}
                        - Perfil: Core {row['pct_core']:.0%}, Sub {row['pct_substitute']:.0%}, Marg {row['pct_marginal']:.0%}
                        - RS: {row['rs_score']:.3f} | MC_avg: {row['avg_mc_score']:.3f}
                        - Precio: ${row['price_mean']:.0f} | Variedad: {row['main_variety']}
                        """)

        with tab3:
            st.header("Análisis a Nivel de Vino")

            # Selector de vinos de la bodega focal
            focal_wines = df_wines[df_wines['winery'] == target_winery]

            st.subheader(f"Vinos de {target_winery}")
            st.dataframe(
                focal_wines[['designation', 'variety', 'points', 'price', 'mc_cluster']].rename(
                    columns={'mc_cluster': 'Cluster MC'}
                ),
                use_container_width=True
            )

            # Comparar con vino específico de competidor
            st.subheader("Comparar Vinos")

            comp_select = st.selectbox("Seleccionar competidor",
                                       df_profiles['competitor'].head(20).tolist())

            if comp_select:
                comp_wines = df_wines[df_wines['winery'] == comp_select]
                st.write(f"**Vinos de {comp_select}:**")
                st.dataframe(
                    comp_wines[['designation', 'variety', 'points', 'price', 'mc_cluster']].rename(
                        columns={'mc_cluster': 'Cluster MC'}
                    ),
                    use_container_width=True
                )

        with tab4:
            st.header("Visualizaciones")

            # Terreno competitivo
            st.subheader("Terreno Competitivo Agregado")

            colors_map = {'Core': '#e74c3c', 'Substitute': '#f39c12',
                         'Marginal': '#3498db', 'Potential': '#95a5a6'}

            fig_terrain = px.scatter(
                df_profiles,
                x='rs_score',
                y='avg_mc_score',
                color='modal_category',
                color_discrete_map=colors_map,
                size='intensity',
                hover_name='competitor',
                hover_data=['pct_core', 'pct_substitute', 'dispersion'],
                labels={'rs_score': 'Resource Similarity (RS)',
                       'avg_mc_score': 'Market Commonality promedio (MC)',
                       'modal_category': 'Categoría'}
            )

            # Añadir líneas de umbral
            mc_thresh = st.session_state.get('mc_thresh', 0.5)
            rs_thresh = st.session_state.get('rs_thresh', 0.5)

            fig_terrain.add_hline(y=mc_thresh, line_dash="dash", line_color="gray", opacity=0.5)
            fig_terrain.add_vline(x=rs_thresh, line_dash="dash", line_color="gray", opacity=0.5)

            st.plotly_chart(fig_terrain, use_container_width=True)

            # Heatmap de perfiles
            st.subheader("Heatmap de Perfiles de Categorías")

            top20 = df_profiles.head(20)
            heatmap_data = top20[['pct_core', 'pct_substitute', 'pct_marginal', 'pct_potential']].values

            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=['Core', 'Substitute', 'Marginal', 'Potential'],
                y=top20['competitor'].tolist(),
                colorscale='YlOrRd',
                text=np.round(heatmap_data * 100, 0),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))

            fig_heat.update_layout(height=600)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Dispersión vs Intensidad
            st.subheader("Dispersión vs Intensidad")

            fig_disp = px.scatter(
                df_profiles,
                x='dispersion',
                y='intensity',
                color='modal_category',
                color_discrete_map=colors_map,
                hover_name='competitor',
                labels={'dispersion': 'Dispersión Competitiva',
                       'intensity': 'Intensidad Competitiva'}
            )

            fig_disp.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)

            st.plotly_chart(fig_disp, use_container_width=True)

    else:
        st.info("👈 Selecciona una bodega y haz clic en 'Ejecutar Análisis Multinivel' para comenzar.")

        # Mostrar explicación del método
        with st.expander("📖 ¿Cómo funciona el Análisis Multinivel?"):
            st.markdown("""
            ### Enfoque Híbrido

            Este análisis implementa un enfoque **multinivel** para la identificación de competidores:

            | Dimensión | Unidad de Análisis | Justificación |
            |-----------|-------------------|---------------|
            | **Market Commonality (MC)** | VINO | Los consumidores eligen vinos, no bodegas |
            | **Resource Similarity (RS)** | BODEGA | Los recursos pertenecen a la empresa |

            ### ABSA (Aspect-Based Sentiment Analysis)

            Extrae sentimientos específicos por aspecto de las notas de cata:

            **Aspectos MC (experiencia del consumidor):**
            - Aroma, Sabor, Final, Valor

            **Aspectos RS (capacidades productivas):**
            - Estructura, Taninos, Acidez, Crianza

            ### Categorización Kamensky

            Cada par de vinos se categoriza según:

            | MC \\ RS | Alto | Bajo |
            |----------|------|------|
            | **Alto** | Core | Substitute |
            | **Bajo** | Marginal | Potential |

            ### Perfil de Categorías

            En lugar de una única categoría por par de bodegas, se genera una **distribución**
            que captura la heterogeneidad de relaciones entre productos.
            """)


if __name__ == "__main__":
    main()
