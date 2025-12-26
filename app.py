"""
Dashboard Interactivo: Análisis de Competidores con Deep Soft Clustering + Kamensky
====================================================================================
Dashboard Streamlit para visualizar y explorar competidores de bodegas españolas.

MEJORAS IMPLEMENTADAS:
- Filtro por variedad de uva
- Filtro por rango de precios
- Búsqueda de bodegas
- Comparador de bodegas
- Radar chart de perfil
- Gráfico de burbujas
- Network graph de competidores
- Score de amenaza
- Análisis de gaps de mercado
- Exportar informe PDF
- Modo oscuro

Para ejecutar:
    streamlit run dashboard_kamensky.py

Autor: Antonio R. Ramos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import ssl
import warnings
import io
import base64
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
# CONFIGURACIÓN DE PÁGINA Y TEMA
# =============================================================================
st.set_page_config(
    page_title="ENOLYTICS by UCA Teams - Análisis de Competidores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODELOS DE DEEP LEARNING
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=[128, 64, 32]):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x):
        return self.encoder(x)


class DeepSoftClustering(nn.Module):
    def __init__(self, input_dim, n_clusters, latent_dim=10, alpha=1.0):
        super(DeepSoftClustering, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.autoencoder = Autoencoder(input_dim, latent_dim)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        x_reconstructed, z = self.autoencoder(x)
        q = self.soft_assignment(z)
        return x_reconstructed, z, q

    def soft_assignment(self, z):
        z_expanded = z.unsqueeze(1)
        centers_expanded = self.cluster_centers.unsqueeze(0)
        dist_sq = torch.sum((z_expanded - centers_expanded) ** 2, dim=2)
        q = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def encode(self, x):
        return self.autoencoder.encode(x)


# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================

@st.cache_data
def get_available_countries(data_path):
    """Obtiene la lista de países disponibles en el dataset."""
    df = pd.read_csv(data_path, usecols=['country'])
    country_counts = df['country'].value_counts()
    return country_counts.index.tolist()


@st.cache_data
def get_available_regions(data_path, country_filter):
    """Obtiene la lista de regiones disponibles para un país específico."""
    df = pd.read_csv(data_path, usecols=['country', 'region_1'])
    df = df[df['country'] == country_filter]
    region_counts = df['region_1'].value_counts()
    region_list = region_counts.index.tolist()
    return ['Todas'] + region_list


@st.cache_data
def get_available_varieties(data_path, country_filter, region_filter='Todas'):
    """Obtiene la lista de variedades disponibles."""
    df = pd.read_csv(data_path, usecols=['country', 'region_1', 'variety'])
    df = df[df['country'] == country_filter]
    if region_filter != 'Todas':
        df = df[df['region_1'] == region_filter]
    variety_counts = df['variety'].value_counts()
    variety_list = variety_counts.index.tolist()
    return ['Todas'] + variety_list


@st.cache_data
def get_price_range(data_path, country_filter):
    """Obtiene el rango de precios para un país."""
    df = pd.read_csv(data_path, usecols=['country', 'price'])
    df = df[df['country'] == country_filter]
    min_price = df['price'].min()
    max_price = df['price'].max()
    return float(min_price) if pd.notna(min_price) else 0.0, float(max_price) if pd.notna(max_price) else 500.0


@st.cache_data
def load_and_process_data(data_path, country_filter='Spain', region_filter='Todas',
                          variety_filter='Todas', price_range=None, min_wines=3):
    """Carga y procesa los datos de vinos con filtros avanzados."""
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw[df_raw['country'] == country_filter].copy()

    # Filtrar por región
    if region_filter != 'Todas':
        df_raw = df_raw[df_raw['region_1'] == region_filter].copy()

    # Filtrar por variedad
    if variety_filter != 'Todas':
        df_raw = df_raw[df_raw['variety'] == variety_filter].copy()

    # Filtrar por rango de precios
    if price_range is not None:
        df_raw = df_raw[(df_raw['price'] >= price_range[0]) & (df_raw['price'] <= price_range[1])].copy()

    # Eliminar duplicados
    n_before = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset=['designation', 'winery'], keep='first')
    df_raw = df_raw.drop_duplicates(subset=['designation', 'variety', 'price', 'points'], keep='first')

    # Sentimientos
    sia = SentimentIntensityAnalyzer()
    sentiments = df_raw['description'].apply(
        lambda x: sia.polarity_scores(str(x)) if pd.notna(x) else {'compound': 0}
    )
    df_raw['sentiment'] = sentiments.apply(lambda x: x['compound'])

    # Agregar a nivel de bodega
    winery_agg = df_raw.groupby('winery').agg({
        'points': ['mean', 'std', 'min', 'max', 'count'],
        'price': ['mean', 'std', 'min', 'max'],
        'sentiment': ['mean', 'std'],
        'variety': lambda x: list(x.dropna().unique()),
        'province': lambda x: list(x.dropna().unique()),
    }).reset_index()

    winery_agg.columns = ['winery', 'avg_points', 'std_points', 'min_points', 'max_points', 'n_wines',
                          'avg_price', 'std_price', 'min_price', 'max_price',
                          'avg_sentiment', 'std_sentiment', 'varieties', 'provinces']

    winery_agg = winery_agg[winery_agg['n_wines'] >= min_wines].copy()

    for col in ['std_points', 'std_price', 'std_sentiment']:
        winery_agg[col] = winery_agg[col].fillna(0)
    winery_agg['avg_price'] = winery_agg['avg_price'].fillna(winery_agg['avg_price'].median())

    winery_agg['n_varieties'] = winery_agg['varieties'].apply(len)
    winery_agg['n_provinces'] = winery_agg['provinces'].apply(len)
    winery_agg['price_range'] = winery_agg['max_price'].fillna(0) - winery_agg['min_price'].fillna(0)
    winery_agg['points_range'] = winery_agg['max_points'] - winery_agg['min_points']

    def get_main_province(winery):
        wines = df_raw[df_raw['winery'] == winery]
        if len(wines) > 0 and len(wines['province'].mode()) > 0:
            return wines['province'].mode().iloc[0]
        return 'Unknown'

    def get_main_variety(winery):
        wines = df_raw[df_raw['winery'] == winery]
        if len(wines) > 0 and len(wines['variety'].mode()) > 0:
            return wines['variety'].mode().iloc[0]
        return 'Unknown'

    winery_agg['main_province'] = winery_agg['winery'].apply(get_main_province)
    winery_agg['main_variety'] = winery_agg['winery'].apply(get_main_variety)

    # Eliminar columnas con listas
    winery_agg = winery_agg.drop(columns=['varieties', 'provinces'])

    return winery_agg.reset_index(drop=True), df_raw


@st.cache_data
def prepare_features(df_wineries):
    """Prepara las features para el modelo."""
    le_province = LabelEncoder()
    le_variety = LabelEncoder()

    df = df_wineries.copy()
    df['province_encoded'] = le_province.fit_transform(df['main_province'])
    df['variety_encoded'] = le_variety.fit_transform(df['main_variety'])

    max_prov = df['province_encoded'].max()
    max_var = df['variety_encoded'].max()
    df['province_encoded'] = df['province_encoded'] / max_prov if max_prov > 0 else 0
    df['variety_encoded'] = df['variety_encoded'] / max_var if max_var > 0 else 0

    mc_features = ['avg_points', 'avg_price', 'province_encoded', 'avg_sentiment', 'std_points']
    rs_features = ['variety_encoded', 'n_varieties', 'n_wines', 'price_range',
                   'points_range', 'n_provinces', 'std_sentiment']

    all_features = mc_features + rs_features

    X = df[all_features].copy().fillna(df[all_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_mc = df[mc_features].copy().fillna(df[mc_features].median())
    X_rs = df[rs_features].copy().fillna(df[rs_features].median())

    scaler_mc = StandardScaler()
    scaler_rs = StandardScaler()

    X_mc_scaled = scaler_mc.fit_transform(X_mc)
    X_rs_scaled = scaler_rs.fit_transform(X_rs)

    return df, X_scaled, X_mc_scaled, X_rs_scaled, all_features


@st.cache_resource
def train_model(X_scaled, n_clusters=7, latent_dim=10, n_epochs=50):
    """Entrena el modelo de Deep Soft Clustering."""
    device = torch.device('cpu')

    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X_scaled.shape[1]
    model = DeepSoftClustering(input_dim, n_clusters, latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for batch_x, in data_loader:
            x_recon, z, q = model(batch_x)
            loss = mse_loss(x_recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(X_tensor).numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(z)
    model.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_)

    with torch.no_grad():
        soft_assignments = model(X_tensor)[2].numpy()

    return model, soft_assignments


def compute_kamensky_analysis(df_wineries, X_mc_scaled, X_rs_scaled, soft_assignments, target_winery):
    """Calcula el análisis de Kamensky para una bodega objetivo."""
    df = df_wineries.copy()

    target_idx = df[df['winery'] == target_winery].index[0]
    target_probs = soft_assignments[target_idx]

    similarities = cosine_similarity([target_probs], soft_assignments)[0]
    df['competition_score'] = similarities

    target_mc = X_mc_scaled[target_idx].reshape(1, -1)
    target_rs = X_rs_scaled[target_idx].reshape(1, -1)

    mc_similarities = cosine_similarity(target_mc, X_mc_scaled)[0]
    rs_similarities = cosine_similarity(target_rs, X_rs_scaled)[0]

    mc_min, mc_max = mc_similarities.min(), mc_similarities.max()
    rs_min, rs_max = rs_similarities.min(), rs_similarities.max()

    df['market_commonality'] = (mc_similarities - mc_min) / (mc_max - mc_min + 1e-10)
    df['resource_similarity'] = (rs_similarities - rs_min) / (rs_max - rs_min + 1e-10)

    mc_threshold = df[df['winery'] != target_winery]['market_commonality'].median()
    rs_threshold = df[df['winery'] != target_winery]['resource_similarity'].median()

    def categorize(row):
        if row['winery'] == target_winery:
            return 'Focal'
        mc_high = row['market_commonality'] >= mc_threshold
        rs_high = row['resource_similarity'] >= rs_threshold
        if mc_high and rs_high:
            return 'Core'
        elif mc_high and not rs_high:
            return 'Substitute'
        elif not mc_high and rs_high:
            return 'Marginal'
        else:
            return 'Potential'

    df['kamensky_category'] = df.apply(categorize, axis=1)

    for i in range(soft_assignments.shape[1]):
        df[f'prob_cluster_{i}'] = soft_assignments[:, i]

    df['cluster'] = np.argmax(soft_assignments, axis=1)

    # Calcular score de amenaza combinado
    df['threat_score'] = calculate_threat_score(df, target_winery)

    return df, mc_threshold, rs_threshold


def calculate_threat_score(df, target_winery):
    """Calcula un score de amenaza combinado para cada competidor."""
    threat_scores = []

    for idx, row in df.iterrows():
        if row['winery'] == target_winery:
            threat_scores.append(0)
            continue

        # Componentes del score de amenaza
        mc_weight = 0.3
        rs_weight = 0.2
        competition_weight = 0.25
        category_weight = 0.25

        category_scores = {'Core': 1.0, 'Substitute': 0.7, 'Marginal': 0.5, 'Potential': 0.2}

        mc_score = row['market_commonality']
        rs_score = row['resource_similarity']
        competition_score = row['competition_score']
        category_score = category_scores.get(row['kamensky_category'], 0.2)

        threat = (mc_weight * mc_score +
                  rs_weight * rs_score +
                  competition_weight * competition_score +
                  category_weight * category_score)

        threat_scores.append(threat)

    return threat_scores


def find_market_gaps(df_analysis, df_raw, target_winery):
    """Identifica nichos de mercado sin competencia."""
    competitors = df_analysis[df_analysis['winery'] != target_winery]
    target_data = df_analysis[df_analysis['winery'] == target_winery].iloc[0]

    gaps = []

    # Analizar gaps por rango de precio
    price_ranges = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 500)]
    for low, high in price_ranges:
        in_range = competitors[(competitors['avg_price'] >= low) & (competitors['avg_price'] < high)]
        if len(in_range) < 3:
            gaps.append({
                'type': 'Precio',
                'description': f'Segmento ${low}-${high}',
                'competitors': len(in_range),
                'opportunity': 'Alta' if len(in_range) == 0 else 'Media'
            })

    # Analizar gaps por puntuación
    points_ranges = [(80, 85), (85, 88), (88, 91), (91, 95), (95, 100)]
    for low, high in points_ranges:
        in_range = competitors[(competitors['avg_points'] >= low) & (competitors['avg_points'] < high)]
        if len(in_range) < 3:
            gaps.append({
                'type': 'Calidad',
                'description': f'Puntuación {low}-{high}',
                'competitors': len(in_range),
                'opportunity': 'Alta' if len(in_range) == 0 else 'Media'
            })

    # Analizar gaps por provincia
    province_counts = competitors['main_province'].value_counts()
    for province in df_raw['province'].dropna().unique():
        count = province_counts.get(province, 0)
        if count < 2:
            gaps.append({
                'type': 'Región',
                'description': province,
                'competitors': count,
                'opportunity': 'Alta' if count == 0 else 'Media'
            })

    return pd.DataFrame(gaps)


# =============================================================================
# FUNCIONES DE INTERPRETACIÓN
# =============================================================================

def generate_profile_interpretation(target_data, competitors):
    """Genera interpretación del perfil de la bodega focal."""
    avg_price_market = competitors['avg_price'].mean()
    avg_points_market = competitors['avg_points'].mean()

    price_position = "premium" if target_data['avg_price'] > avg_price_market * 1.2 else \
                     "económico" if target_data['avg_price'] < avg_price_market * 0.8 else "medio"

    quality_position = "alta calidad" if target_data['avg_points'] > avg_points_market + 2 else \
                       "calidad estándar" if target_data['avg_points'] > avg_points_market - 2 else "calidad inferior"

    sentiment_level = "muy positivo" if target_data['avg_sentiment'] > 0.5 else \
                      "positivo" if target_data['avg_sentiment'] > 0.3 else \
                      "neutral" if target_data['avg_sentiment'] > 0 else "negativo"

    interpretation = f"""
**Posicionamiento de {target_data['winery']}:**

- **Segmento de precio**: La bodega opera en el segmento **{price_position}** con un precio medio de ${target_data['avg_price']:.2f}
  (media del mercado: ${avg_price_market:.2f}).

- **Calidad percibida**: Con {target_data['avg_points']:.1f} puntos, la bodega se posiciona en el segmento de **{quality_position}**
  (media del mercado: {avg_points_market:.1f} puntos).

- **Percepción del consumidor**: El sentimiento en las reseñas es **{sentiment_level}** ({target_data['avg_sentiment']:.3f}).

- **Diversificación**: Con {int(target_data['n_varieties'])} variedades y {int(target_data['n_wines'])} vinos,
  {'la bodega tiene un portafolio diversificado' if target_data['n_varieties'] > 3 else 'la bodega está especializada'}.
"""
    return interpretation


def generate_kamensky_matrix_interpretation(competitors, target_winery):
    """Genera interpretación de la matriz de Kamensky."""
    core_count = len(competitors[competitors['kamensky_category'] == 'Core'])
    substitute_count = len(competitors[competitors['kamensky_category'] == 'Substitute'])
    marginal_count = len(competitors[competitors['kamensky_category'] == 'Marginal'])
    potential_count = len(competitors[competitors['kamensky_category'] == 'Potential'])
    total = len(competitors)

    core_pct = core_count / total * 100 if total > 0 else 0

    if core_pct > 35:
        competitive_situation = "ALTA INTENSIDAD COMPETITIVA"
        competitive_desc = "La bodega enfrenta un entorno altamente competitivo."
    elif core_pct > 25:
        competitive_situation = "INTENSIDAD COMPETITIVA MODERADA"
        competitive_desc = "Existe un nivel equilibrado de competencia directa."
    else:
        competitive_situation = "BAJA INTENSIDAD COMPETITIVA DIRECTA"
        competitive_desc = "La bodega tiene pocos competidores directos."

    top_core = competitors[competitors['kamensky_category'] == 'Core'].nlargest(3, 'competition_score')
    top_core_names = ", ".join(top_core['winery'].values) if len(top_core) > 0 else "N/A"

    interpretation = f"""
**Situación: {competitive_situation}**

{competitive_desc}

**Distribución de competidores:**
- **CORE ({core_count}, {core_pct:.1f}%)**: Competidores directos. Top: {top_core_names}
- **SUBSTITUTE ({substitute_count}, {substitute_count/total*100:.1f}%)**: Amenaza de sustitución
- **MARGINAL ({marginal_count}, {marginal_count/total*100:.1f}%)**: Amenaza latente
- **POTENTIAL ({potential_count}, {potential_count/total*100:.1f}%)**: Vigilar movimientos
"""
    return interpretation


def generate_strategic_diagnosis(target_data, competitors, mc_threshold, rs_threshold):
    """Genera diagnóstico estratégico y recomendaciones finales."""
    core_count = len(competitors[competitors['kamensky_category'] == 'Core'])
    substitute_count = len(competitors[competitors['kamensky_category'] == 'Substitute'])
    marginal_count = len(competitors[competitors['kamensky_category'] == 'Marginal'])
    total = len(competitors)

    core_pct = core_count / total * 100 if total > 0 else 0

    avg_price = competitors['avg_price'].mean()
    avg_points = competitors['avg_points'].mean()

    price_advantage = target_data['avg_price'] < avg_price * 0.9
    quality_advantage = target_data['avg_points'] > avg_points + 1
    sentiment_advantage = target_data['avg_sentiment'] > competitors['avg_sentiment'].mean()

    top_core = competitors[competitors['kamensky_category'] == 'Core'].nlargest(3, 'competition_score')

    diagnosis = f"""
## DIAGNÓSTICO ESTRATÉGICO: {target_data['winery']}

### 1. POSICIÓN COMPETITIVA ACTUAL

| Dimensión | {target_data['winery']} | Media Mercado | Ventaja |
|-----------|------------------------|---------------|---------|
| Precio | ${target_data['avg_price']:.2f} | ${avg_price:.2f} | {'Sí' if price_advantage else 'No'} |
| Calidad | {target_data['avg_points']:.1f} pts | {avg_points:.1f} pts | {'Sí' if quality_advantage else 'No'} |
| Sentimiento | {target_data['avg_sentiment']:.3f} | {competitors['avg_sentiment'].mean():.3f} | {'Sí' if sentiment_advantage else 'No'} |

### 2. INTENSIDAD COMPETITIVA

- **Competidores Core**: {core_count} ({core_pct:.1f}%) - {'ALTA PRESIÓN' if core_pct > 30 else 'Manejable'}
- **Amenaza de Sustitutos**: {substitute_count} ({substitute_count/total*100:.1f}%)
- **Riesgo de Nuevas Entradas**: {marginal_count} ({marginal_count/total*100:.1f}%)

### 3. PRINCIPALES AMENAZAS
"""

    for i, (_, row) in enumerate(top_core.iterrows(), 1):
        diagnosis += f"""
{i}. **{row['winery']}** (Score: {row['competition_score']:.3f})
   - Precio: ${row['avg_price']:.2f} | Calidad: {row['avg_points']:.1f} pts
"""

    diagnosis += """
### 4. RECOMENDACIONES ESTRATÉGICAS
"""

    if core_pct > 30:
        diagnosis += """
**DIFERENCIACIÓN URGENTE** - Alta concentración de competidores Core.
"""
    if not quality_advantage and not price_advantage:
        diagnosis += """
**MEJORA DE PROPUESTA DE VALOR** - No hay ventaja clara en precio ni calidad.
"""
    if quality_advantage:
        diagnosis += """
**ESTRATEGIA PREMIUM** - Aprovechar ventaja en calidad.
"""
    if price_advantage:
        diagnosis += """
**LIDERAZGO EN COSTOS** - Mantener ventaja en precio.
"""
    if not sentiment_advantage:
        diagnosis += """
**MEJORA DE PERCEPCIÓN** - Trabajar en reputación online.
"""

    return diagnosis


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_kamensky_matrix(df, target_winery, mc_threshold, rs_threshold):
    """Genera la matriz de Kamensky interactiva."""
    competitors = df[df['winery'] != target_winery].copy()

    color_map = {
        'Core': '#e74c3c',
        'Substitute': '#f39c12',
        'Marginal': '#3498db',
        'Potential': '#95a5a6'
    }

    fig = px.scatter(
        competitors,
        x='resource_similarity',
        y='market_commonality',
        color='kamensky_category',
        size='threat_score',
        color_discrete_map=color_map,
        custom_data=['winery'],
        title=f'Matriz de Kamensky - Competidores de {target_winery}',
        labels={
            'resource_similarity': 'Resource Similarity (RS)',
            'market_commonality': 'Market Commonality (MC)',
            'kamensky_category': 'Categoría',
            'threat_score': 'Score Amenaza'
        }
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

    fig.add_hline(y=mc_threshold, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=rs_threshold, line_dash="dash", line_color="black", opacity=0.5)

    fig.add_annotation(x=0.85, y=0.95, text="CORE", showarrow=False,
                       font=dict(size=14, color='#e74c3c', weight='bold'), xref='paper', yref='paper')
    fig.add_annotation(x=0.15, y=0.95, text="SUBSTITUTE", showarrow=False,
                       font=dict(size=14, color='#f39c12', weight='bold'), xref='paper', yref='paper')
    fig.add_annotation(x=0.85, y=0.05, text="MARGINAL", showarrow=False,
                       font=dict(size=14, color='#3498db', weight='bold'), xref='paper', yref='paper')
    fig.add_annotation(x=0.15, y=0.05, text="POTENTIAL", showarrow=False,
                       font=dict(size=14, color='#95a5a6', weight='bold'), xref='paper', yref='paper')

    fig.update_layout(height=500, xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]))

    return fig


def plot_category_distribution(df, target_winery):
    """Gráfico de distribución por categoría."""
    competitors = df[df['winery'] != target_winery]
    counts = competitors['kamensky_category'].value_counts()

    color_map = {'Core': '#e74c3c', 'Substitute': '#f39c12', 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    fig = go.Figure(data=[
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[color_map.get(cat, '#333') for cat in counts.index],
            text=[f'{v} ({v/len(competitors)*100:.1f}%)' for v in counts.values],
            textposition='auto'
        )
    ])

    fig.update_layout(title='Distribución por Categoría Kamensky', xaxis_title='Categoría',
                      yaxis_title='Número de Bodegas', height=400)
    return fig


def plot_radar_chart(target_data, competitors):
    """Genera un radar chart del perfil de la bodega."""
    categories = ['Precio', 'Calidad', 'Sentimiento', 'Diversidad', 'Volumen']

    # Normalizar valores
    max_price = competitors['avg_price'].max()
    max_points = 100
    max_sentiment = 1
    max_varieties = competitors['n_varieties'].max()
    max_wines = competitors['n_wines'].max()

    target_values = [
        target_data['avg_price'] / max_price if max_price > 0 else 0,
        target_data['avg_points'] / max_points,
        (target_data['avg_sentiment'] + 1) / 2,  # Normalizar de -1,1 a 0,1
        target_data['n_varieties'] / max_varieties if max_varieties > 0 else 0,
        target_data['n_wines'] / max_wines if max_wines > 0 else 0
    ]

    market_avg = [
        competitors['avg_price'].mean() / max_price if max_price > 0 else 0,
        competitors['avg_points'].mean() / max_points,
        (competitors['avg_sentiment'].mean() + 1) / 2,
        competitors['n_varieties'].mean() / max_varieties if max_varieties > 0 else 0,
        competitors['n_wines'].mean() / max_wines if max_wines > 0 else 0
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=target_values + [target_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=target_data['winery'],
        line_color='#e74c3c'
    ))

    fig.add_trace(go.Scatterpolar(
        r=market_avg + [market_avg[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Media Mercado',
        line_color='#3498db',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f'Perfil Multidimensional: {target_data["winery"]}',
        height=450
    )

    return fig


def plot_bubble_chart(df, target_winery):
    """Genera un gráfico de burbujas."""
    competitors = df[df['winery'] != target_winery].copy()

    color_map = {'Core': '#e74c3c', 'Substitute': '#f39c12', 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    fig = px.scatter(
        competitors,
        x='avg_price',
        y='avg_points',
        size='n_wines',
        color='kamensky_category',
        color_discrete_map=color_map,
        custom_data=['winery'],
        title='Mapa de Mercado: Precio vs Calidad (tamaño = nº vinos)',
        labels={
            'avg_price': 'Precio Promedio ($)',
            'avg_points': 'Puntuación Promedio',
            'n_wines': 'Nº Vinos',
            'kamensky_category': 'Categoría'
        }
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

    # Añadir bodega focal
    target_data = df[df['winery'] == target_winery].iloc[0]
    fig.add_trace(go.Scatter(
        x=[target_data['avg_price']],
        y=[target_data['avg_points']],
        mode='markers+text',
        marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
        text=[target_winery],
        textposition='top center',
        name='Bodega Focal',
        showlegend=True,
        hovertemplate=f'{target_winery}<extra></extra>'
    ))

    fig.update_layout(height=500)
    return fig


def plot_network_graph(df, target_winery, n_connections=15):
    """Genera un grafo de red de competidores."""
    competitors = df[df['winery'] != target_winery].nlargest(n_connections, 'competition_score')
    target_data = df[df['winery'] == target_winery].iloc[0]

    # Crear grafo
    G = nx.Graph()

    # Añadir nodo focal
    G.add_node(target_winery, category='Focal', size=30)

    # Añadir competidores
    for _, row in competitors.iterrows():
        G.add_node(row['winery'], category=row['kamensky_category'],
                   size=row['competition_score'] * 20)
        G.add_edge(target_winery, row['winery'], weight=row['competition_score'])

    # Obtener posiciones usando spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Crear trazas para Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    color_map = {'Focal': 'gold', 'Core': '#e74c3c', 'Substitute': '#f39c12',
                 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        category = G.nodes[node]['category']
        node_color.append(color_map.get(category, '#333'))
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertemplate='%{text}<extra></extra>',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Red de Competidores: {target_winery}',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))

    return fig


def plot_threat_ranking(df, target_winery, n_top=10):
    """Genera un ranking de amenazas."""
    competitors = df[df['winery'] != target_winery].nlargest(n_top, 'threat_score')

    color_map = {'Core': '#e74c3c', 'Substitute': '#f39c12', 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    fig = go.Figure(go.Bar(
        x=competitors['threat_score'],
        y=competitors['winery'],
        orientation='h',
        marker_color=[color_map.get(cat, '#333') for cat in competitors['kamensky_category']],
        text=[f"{score:.3f}" for score in competitors['threat_score']],
        textposition='auto',
        hovertemplate='%{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {n_top} Amenazas Competitivas',
        xaxis_title='Score de Amenaza',
        yaxis_title='Bodega',
        height=400,
        yaxis=dict(autorange="reversed")
    )

    return fig


def plot_comparison_chart(df, wineries_to_compare):
    """Genera un gráfico comparativo de múltiples bodegas."""
    if len(wineries_to_compare) < 2:
        return None

    data = df[df['winery'].isin(wineries_to_compare)]

    metrics = ['avg_points', 'avg_price', 'avg_sentiment', 'n_wines', 'n_varieties']
    metric_labels = ['Puntuación', 'Precio ($)', 'Sentimiento', 'Nº Vinos', 'Nº Variedades']

    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metric_labels)

    colors = px.colors.qualitative.Set1[:len(wineries_to_compare)]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels), 1):
        for j, winery in enumerate(wineries_to_compare):
            winery_data = data[data['winery'] == winery]
            if len(winery_data) > 0:
                value = winery_data[metric].values[0]
                fig.add_trace(
                    go.Bar(
                        x=[winery],
                        y=[value],
                        name=winery if i == 1 else '',
                        marker_color=colors[j],
                        showlegend=(i == 1),
                        text=[f'{value:.2f}' if metric != 'n_wines' and metric != 'n_varieties' else f'{int(value)}'],
                        textposition='auto'
                    ),
                    row=1, col=i
                )

    fig.update_layout(height=400, title='Comparativa de Bodegas', barmode='group')
    return fig


def plot_competition_heatmap(df, target_winery, n_top=15):
    """Heatmap de probabilidades de cluster."""
    competitors = df[df['winery'] != target_winery].nlargest(n_top, 'competition_score')

    prob_cols = [col for col in df.columns if col.startswith('prob_cluster_')]
    prob_matrix = competitors[prob_cols].values

    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=[f'C{i}' for i in range(len(prob_cols))],
        y=competitors['winery'].values,
        colorscale='YlOrRd',
        colorbar=dict(title='Probabilidad'),
        hovertemplate='%{y}<extra></extra>'
    ))

    fig.update_layout(title=f'Distribución de Clusters - Top {n_top} Competidores',
                      xaxis_title='Cluster', yaxis_title='Bodega', height=500)
    return fig


def plot_price_vs_mc(df, target_winery, mc_threshold):
    """Gráfico de precio vs Market Commonality."""
    competitors = df[df['winery'] != target_winery]

    color_map = {'Core': '#e74c3c', 'Substitute': '#f39c12', 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    fig = px.scatter(
        competitors,
        x='avg_price',
        y='market_commonality',
        color='kamensky_category',
        color_discrete_map=color_map,
        custom_data=['winery'],
        title='Market Commonality vs Precio',
        labels={'avg_price': 'Precio Promedio ($)', 'market_commonality': 'Market Commonality'}
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

    fig.add_hline(y=mc_threshold, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(height=400)
    return fig


def plot_rs_boxplot(df, target_winery, rs_threshold):
    """Boxplot de RS por categoría."""
    competitors = df[df['winery'] != target_winery]

    color_map = {'Core': '#e74c3c', 'Substitute': '#f39c12', 'Marginal': '#3498db', 'Potential': '#95a5a6'}

    fig = go.Figure()

    for category in ['Core', 'Substitute', 'Marginal', 'Potential']:
        cat_data = competitors[competitors['kamensky_category'] == category]['resource_similarity']
        if len(cat_data) > 0:
            fig.add_trace(go.Box(y=cat_data, name=category, marker_color=color_map[category]))

    fig.add_hline(y=rs_threshold, line_dash="dash", line_color="black", opacity=0.5,
                  annotation_text=f"Umbral RS: {rs_threshold:.3f}")

    fig.update_layout(title='Resource Similarity por Categoría', yaxis_title='Resource Similarity', height=400)
    return fig


# =============================================================================
# FUNCIONES DE EXPORTACIÓN
# =============================================================================

def generate_pdf_report(target_data, competitors, df_analysis, target_winery,
                        mc_threshold, rs_threshold, gaps_df):
    """Genera un informe en formato texto (simula PDF)."""
    report = f"""
================================================================================
                    INFORME DE ANÁLISIS COMPETITIVO
                    {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

BODEGA FOCAL: {target_winery}
--------------------------------------------------------------------------------

PERFIL DE LA BODEGA
-------------------
- Puntuación Media: {target_data['avg_points']:.1f} puntos
- Precio Medio: ${target_data['avg_price']:.2f}
- Sentimiento: {target_data['avg_sentiment']:.3f}
- Número de Vinos: {int(target_data['n_wines'])}
- Variedades: {int(target_data['n_varieties'])}
- Provincia Principal: {target_data['main_province']}
- Variedad Principal: {target_data['main_variety']}

ANÁLISIS COMPETITIVO KAMENSKY
-----------------------------
Total de competidores analizados: {len(competitors)}

Distribución por categoría:
- Core (competidores directos): {len(competitors[competitors['kamensky_category'] == 'Core'])}
- Substitute (sustitutos): {len(competitors[competitors['kamensky_category'] == 'Substitute'])}
- Marginal (amenaza latente): {len(competitors[competitors['kamensky_category'] == 'Marginal'])}
- Potential (vigilar): {len(competitors[competitors['kamensky_category'] == 'Potential'])}

TOP 5 AMENAZAS
--------------
"""

    top_threats = competitors.nlargest(5, 'threat_score')
    for i, (_, row) in enumerate(top_threats.iterrows(), 1):
        report += f"""
{i}. {row['winery']}
   - Categoría: {row['kamensky_category']}
   - Score de Amenaza: {row['threat_score']:.3f}
   - Precio: ${row['avg_price']:.2f}
   - Puntuación: {row['avg_points']:.1f}
"""

    report += """
OPORTUNIDADES DE MERCADO (GAPS)
-------------------------------
"""
    if len(gaps_df) > 0:
        for _, gap in gaps_df.head(5).iterrows():
            report += f"- {gap['type']}: {gap['description']} (Competidores: {gap['competitors']}, Oportunidad: {gap['opportunity']})\n"
    else:
        report += "No se identificaron gaps significativos.\n"

    report += """
================================================================================
Generado con Dashboard Kamensky - Deep Soft Clustering
================================================================================
"""
    return report


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

def main():
    # Estilos CSS profesionales
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1a1a2e;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .sub-header {
            text-align: center;
            color: #4a4a6a;
            font-weight: 400;
            margin-top: 0;
            padding-top: 0;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #2c3e50;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
        }
        h3 {
            color: #34495e;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>ENOLYTICS by UCA Teams</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Módulo de identificación y análisis de competidores</h3>", unsafe_allow_html=True)

    st.sidebar.header("Configuración")

    # URL de Dropbox para cargar datos (funciona en local y en Streamlit Cloud)
    # Nota: dl=1 fuerza la descarga directa del archivo
    data_path = "https://www.dropbox.com/scl/fi/aslyox4hd6st71mdlb2mc/winemag-data_first150k.csv?rlkey=86hu9701tjcvs2st51dlugzvd&dl=1"

    # Selector de país
    st.sidebar.subheader("País")
    available_countries = get_available_countries(data_path)
    default_country_idx = available_countries.index('Spain') if 'Spain' in available_countries else 0
    selected_country = st.sidebar.selectbox("Selecciona el país:", available_countries, index=default_country_idx)

    # Selector de región
    st.sidebar.subheader("Región")
    available_regions = get_available_regions(data_path, selected_country)
    selected_region = st.sidebar.selectbox("Selecciona la región:", available_regions, index=0)

    # Selector de variedad
    st.sidebar.subheader("Variedad")
    available_varieties = get_available_varieties(data_path, selected_country, selected_region)
    selected_variety = st.sidebar.selectbox("Selecciona la variedad:", available_varieties, index=0)

    # Rango de precios
    st.sidebar.subheader("Rango de Precios")
    min_price, max_price = get_price_range(data_path, selected_country)
    price_range = st.sidebar.slider("Rango de precios ($):",
                                     min_value=int(min_price),
                                     max_value=int(max_price),
                                     value=(int(min_price), int(max_price)))

    # Parámetros
    min_wines = st.sidebar.slider("Mínimo de vinos por bodega", 2, 10, 3)

    # Cargar datos
    region_msg = f" - {selected_region}" if selected_region != 'Todas' else ""
    variety_msg = f" - {selected_variety}" if selected_variety != 'Todas' else ""

    with st.spinner(f"Cargando datos de {selected_country}{region_msg}{variety_msg}..."):
        df_wineries, df_raw = load_and_process_data(
            data_path,
            country_filter=selected_country,
            region_filter=selected_region,
            variety_filter=selected_variety,
            price_range=price_range,
            min_wines=min_wines
        )

    # Verificar si hay suficientes bodegas
    n_wineries = len(df_wineries)
    if n_wineries < 3:
        st.error(f"Solo hay {n_wineries} bodegas en esta selección. Se necesitan al menos 3.")
        st.info("Prueba a ajustar los filtros.")
        st.stop()

    # Ajustar número de clusters
    max_clusters = min(12, n_wineries - 1)
    default_clusters = min(7, max_clusters)
    n_clusters = st.sidebar.slider("Número de clusters", 2, max_clusters, default_clusters)

    # Preparar features
    df_features, X_scaled, X_mc_scaled, X_rs_scaled, feature_names = prepare_features(df_wineries)

    st.sidebar.success(f"{len(df_wineries)} bodegas encontradas")

    winery_list = sorted(df_features['winery'].unique().tolist())

    # Selector de bodega focal
    st.sidebar.subheader("Bodega Focal")
    target_winery = st.sidebar.selectbox("Selecciona la bodega a analizar:", winery_list, index=0)

    # Entrenar modelo
    with st.spinner("Entrenando modelo de Deep Soft Clustering..."):
        model, soft_assignments = train_model(X_scaled, n_clusters=n_clusters)

    # Análisis de Kamensky
    with st.spinner("Calculando análisis de Kamensky..."):
        df_analysis, mc_threshold, rs_threshold = compute_kamensky_analysis(
            df_features, X_mc_scaled, X_rs_scaled, soft_assignments, target_winery
        )

    competitors = df_analysis[df_analysis['winery'] != target_winery]
    target_data = df_analysis[df_analysis['winery'] == target_winery].iloc[0]

    # ==========================================================================
    # PERFIL DE LA BODEGA FOCAL
    # ==========================================================================
    st.header(f"Perfil de {target_winery}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Puntuación Media", f"{target_data['avg_points']:.1f}")
    col2.metric("Precio Medio", f"${target_data['avg_price']:.2f}")
    col3.metric("Sentimiento", f"{target_data['avg_sentiment']:.3f}")
    col4.metric("Nº Vinos", int(target_data['n_wines']))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provincia", target_data['main_province'])
    col2.metric("Variedad Principal", target_data['main_variety'])
    col3.metric("Cluster Principal", int(target_data['cluster']))
    col4.metric("Nº Variedades", int(target_data['n_varieties']))

    # Radar Chart del perfil
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_radar_chart(target_data, competitors), width='stretch')

    with col2:
        with st.expander("Interpretación del Perfil", expanded=True):
            st.markdown(generate_profile_interpretation(target_data, competitors))

    # Catálogo de vinos
    with st.expander("Catálogo de Vinos de la Bodega", expanded=False):
        target_wines = df_raw[df_raw['winery'] == target_winery].copy()
        if len(target_wines) > 0:
            wine_cols = ['designation', 'variety', 'price', 'province', 'points']
            available_cols = [col for col in wine_cols if col in target_wines.columns]
            wines_display = target_wines[available_cols].copy()
            col_rename = {'designation': 'Nombre', 'variety': 'Variedad', 'price': 'Precio ($)',
                          'province': 'Provincia', 'points': 'Puntuación'}
            wines_display = wines_display.rename(columns={k: v for k, v in col_rename.items() if k in wines_display.columns})
            if 'Puntuación' in wines_display.columns:
                wines_display = wines_display.sort_values('Puntuación', ascending=False)
            st.dataframe(wines_display, width='stretch', hide_index=True)

    # ==========================================================================
    # COMPARADOR DE BODEGAS
    # ==========================================================================
    st.header("Comparador de Bodegas")

    wineries_to_compare = st.multiselect(
        "Selecciona bodegas para comparar (2-4):",
        winery_list,
        default=[target_winery],
        max_selections=4
    )

    if len(wineries_to_compare) >= 2:
        comparison_chart = plot_comparison_chart(df_analysis, wineries_to_compare)
        if comparison_chart:
            st.plotly_chart(comparison_chart, width='stretch')
    else:
        st.info("Selecciona al menos 2 bodegas para comparar")

    # ==========================================================================
    # RESUMEN DE COMPETIDORES Y AMENAZAS
    # ==========================================================================
    st.header("Resumen de Competidores")

    col1, col2, col3, col4 = st.columns(4)

    for col, category in zip(
        [col1, col2, col3, col4],
        ['Core', 'Substitute', 'Marginal', 'Potential']
    ):
        count = len(competitors[competitors['kamensky_category'] == category])
        pct = count / len(competitors) * 100 if len(competitors) > 0 else 0
        col.metric(f"{category}", f"{count} ({pct:.1f}%)")

    # Ranking de amenazas
    st.subheader("Ranking de Amenazas")
    st.plotly_chart(plot_threat_ranking(df_analysis, target_winery, 10), width='stretch')

    # ==========================================================================
    # MATRIZ DE KAMENSKY
    # ==========================================================================
    st.header("Matriz de Kamensky")

    fig_kamensky = plot_kamensky_matrix(df_analysis, target_winery, mc_threshold, rs_threshold)
    st.plotly_chart(fig_kamensky, width='stretch')

    with st.expander("Interpretación de la Matriz", expanded=True):
        st.markdown(generate_kamensky_matrix_interpretation(competitors, target_winery))

    # ==========================================================================
    # VISUALIZACIONES AVANZADAS
    # ==========================================================================
    st.header("Análisis Detallado")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Burbujas", "Red", "Distribución", "Clusters", "Precio vs MC", "RS Categoría"
    ])

    with tab1:
        st.plotly_chart(plot_bubble_chart(df_analysis, target_winery), width='stretch')
        st.markdown("""
        **Interpretación**: El tamaño de cada burbuja representa el número de vinos de la bodega.
        La estrella dorada marca la posición de la bodega focal.
        """)

    with tab2:
        n_connections = st.slider("Número de conexiones", 5, 25, 15)
        st.plotly_chart(plot_network_graph(df_analysis, target_winery, n_connections), width='stretch')
        st.markdown("""
        **Interpretación**: La red muestra las conexiones competitivas.
        Nodos más grandes = mayor competencia con la bodega focal.
        """)

    with tab3:
        st.plotly_chart(plot_category_distribution(df_analysis, target_winery), width='stretch')

    with tab4:
        n_top = st.slider("Número de top competidores", 10, 30, 15)
        st.plotly_chart(plot_competition_heatmap(df_analysis, target_winery, n_top), width='stretch')

    with tab5:
        st.plotly_chart(plot_price_vs_mc(df_analysis, target_winery, mc_threshold), width='stretch')

    with tab6:
        st.plotly_chart(plot_rs_boxplot(df_analysis, target_winery, rs_threshold), width='stretch')

    # ==========================================================================
    # ANÁLISIS DE GAPS DE MERCADO
    # ==========================================================================
    st.header("Oportunidades de Mercado (Gap Analysis)")

    gaps_df = find_market_gaps(df_analysis, df_raw, target_winery)

    if len(gaps_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gaps Identificados")
            st.dataframe(gaps_df.head(10), width='stretch', hide_index=True)

        with col2:
            st.subheader("Distribución por Tipo")
            gap_counts = gaps_df['type'].value_counts()
            fig = px.pie(values=gap_counts.values, names=gap_counts.index, title='Tipos de Oportunidades')
            st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **Interpretación**: Los gaps representan nichos de mercado con poca o ninguna competencia.
        - **Alta oportunidad**: 0 competidores en ese segmento
        - **Media oportunidad**: 1-2 competidores en ese segmento
        """)
    else:
        st.info("No se identificaron gaps significativos en el mercado actual.")

    # ==========================================================================
    # TOP COMPETIDORES POR CATEGORÍA
    # ==========================================================================
    st.header("Top Competidores por Categoría")

    category_selected = st.selectbox("Selecciona categoría:", ['Core', 'Substitute', 'Marginal', 'Potential'])

    cat_competitors = competitors[competitors['kamensky_category'] == category_selected].nlargest(10, 'competition_score')

    st.dataframe(
        cat_competitors[['winery', 'competition_score', 'threat_score', 'market_commonality',
                        'resource_similarity', 'avg_points', 'avg_price', 'main_province']].round(3),
        width='stretch',
        hide_index=True
    )

    # ==========================================================================
    # DIAGNÓSTICO ESTRATÉGICO
    # ==========================================================================
    st.header("Diagnóstico Estratégico")

    st.markdown(generate_strategic_diagnosis(target_data, competitors, mc_threshold, rs_threshold))

    # ==========================================================================
    # EXPORTAR DATOS
    # ==========================================================================
    st.header("Exportar Resultados")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_competitors = competitors[['winery', 'kamensky_category', 'competition_score', 'threat_score',
                                       'market_commonality', 'resource_similarity',
                                       'avg_points', 'avg_price', 'main_province', 'main_variety']].to_csv(index=False)
        st.download_button(
            label="Descargar Competidores (CSV)",
            data=csv_competitors,
            file_name=f"competidores_{target_winery}.csv",
            mime="text/csv"
        )

    with col2:
        if len(gaps_df) > 0:
            csv_gaps = gaps_df.to_csv(index=False)
            st.download_button(
                label="Descargar Gaps (CSV)",
                data=csv_gaps,
                file_name=f"gaps_mercado_{target_winery}.csv",
                mime="text/csv"
            )

    with col3:
        report = generate_pdf_report(target_data, competitors, df_analysis, target_winery,
                                     mc_threshold, rs_threshold, gaps_df)
        st.download_button(
            label="Descargar Informe (TXT)",
            data=report,
            file_name=f"informe_{target_winery}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; padding: 20px 0;'>
        <p><strong>ENOLYTICS by UCA Teams</strong> - Sistema de Análisis Competitivo</p>
        <p>Metodología: Deep Soft Clustering + Teoría de Rivalidad Competitiva (Chen, 1996; Kamensky, 2000)</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #999; font-size: 11px;'>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
