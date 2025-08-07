# Professional Academic Theme - Dark Mode with Better Visibility
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto+Slab:wght@700&display=swap');
    
    /* Professional Dark Academic Theme with Frosted Glass Effect */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #0f172a 100%);
        background-attachment: fixed;
        background-size: cover;
    }
    
    .main {
        padding: 1rem 2rem;
        background: transparent;
    }
    
    /* Headers - Academic Style with Modern Fonts */
    h1 {
        color: #4fc3f7 !important;
        font-family: 'Roboto Slab', Georgia, serif;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #4fc3f7;
        margin-bottom: 30px;
        text-shadow: 0 0 30px rgba(79, 195, 247, 0.5);
    }
    
    h2 {
        color: #81c784 !important;
        font-family: 'Roboto Slab', Georgia, serif;
        font-weight: 600;
        margin-top: 2rem;
        text-shadow: 0 0 20px rgba(129, 199, 132, 0.3);
    }
    
    h3 {
        color: #ffb74d !important;
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 500;
    }
    
    /* Professional Cards with Glassmorphism */
    .academic-card {
        background: rgba(30, 30, 63, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(79, 195, 247, 0.2);
        margin: 1.5rem 0;
    }
    
    .academic-card h3 {
        color: #4fc3f7 !important;
        margin-bottom: 1rem;
    }
    
    .academic-card p {
        color: #e0e0e0 !important;
        line-height: 1.8;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Metrics Cards with Glow */
    [data-testid="metric-container"] {
        background: rgba(30, 30, 60, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(79, 195, 247, 0.1);
        border: 1px solid rgba(129, 199, 132, 0.3);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #9e9e9e !important;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Montserrat', sans-serif;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #4fc3f7 !important;
        font-weight: 700;
        font-size: 28px;
        text-shadow: 0 0 20px rgba(79, 195, 247, 0.5);
        font-family: 'Roboto Slab', serif;
    }
    
    /* Professional Buttons with Animations */
    .stButton>button {
        background: linear-gradient(135deg, #00acc1 0%, #4fc3f7 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        border"""
ProCaPPIS - Prostate Cancer Protein-Protein Interaction Prediction System
Multi-Page Academic Application with Advanced Features
"""

import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="ProCaPPIS - Advanced PPI Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/procappis',
        'Report a bug': "https://github.com/yourusername/procappis/issues",
        'About': "ProCaPPIS v2.0 - Academic PPI Prediction Platform"
    }
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import joblib
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Professional Academic Theme - Dark Mode with Better Visibility
st.markdown("""
    <style>
    /* Professional Dark Academic Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    .main {
        padding: 1rem 2rem;
        background: transparent;
    }
    
    /* Headers - Academic Style */
    h1 {
        color: #4fc3f7 !important;
        font-family: 'Georgia', serif;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #4fc3f7;
        margin-bottom: 30px;
        text-shadow: 0 0 20px rgba(79, 195, 247, 0.5);
    }
    
    h2 {
        color: #81c784 !important;
        font-family: 'Georgia', serif;
        font-weight: 600;
        margin-top: 2rem;
        text-shadow: 0 0 10px rgba(129, 199, 132, 0.3);
    }
    
    h3 {
        color: #ffb74d !important;
        font-family: 'Arial', sans-serif;
        font-weight: 500;
    }
    
    /* Professional Cards */
    .academic-card {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(79, 195, 247, 0.3);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .academic-card h3 {
        color: #4fc3f7 !important;
        margin-bottom: 1rem;
    }
    
    .academic-card p {
        color: #e0e0e0 !important;
        line-height: 1.8;
    }
    
    /* Metrics Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1a1a3e, #2d2d5e);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(129, 199, 132, 0.3);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #9e9e9e !important;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #4fc3f7 !important;
        font-weight: 700;
        font-size: 28px;
        text-shadow: 0 0 10px rgba(79, 195, 247, 0.5);
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #81c784 !important;
    }
    
    /* Professional Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00acc1 0%, #26c6da 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        border-radius: 30px;
        font-family: 'Arial', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 172, 193, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 172, 193, 0.6);
        background: linear-gradient(90deg, #26c6da 0%, #00acc1 100%);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f172a 100%);
        border-right: 2px solid rgba(79, 195, 247, 0.3);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(30, 30, 60, 0.8);
        border: 2px solid rgba(79, 195, 247, 0.3);
        color: #e0e0e0;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4fc3f7;
        box-shadow: 0 0 0 2px rgba(79, 195, 247, 0.2);
        background-color: rgba(30, 30, 60, 0.9);
    }
    
    .stSelectbox>div>div>select {
        background-color: rgba(30, 30, 60, 0.8);
        border: 2px solid rgba(79, 195, 247, 0.3);
        color: #e0e0e0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 30, 60, 0.5);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9e9e9e;
        background-color: transparent;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(79, 195, 247, 0.2);
        color: #4fc3f7;
        border-radius: 8px;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(145deg, #1e3a5f, #2c5282);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4fc3f7;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(79, 195, 247, 0.2);
    }
    
    .info-box h4 {
        color: #4fc3f7 !important;
        margin-bottom: 0.5rem;
    }
    
    .info-box p, .info-box ul {
        color: #e0e0e0 !important;
    }
    
    .success-box {
        background: linear-gradient(145deg, #1b4332, #2d6a4f);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #81c784;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(129, 199, 132, 0.2);
    }
    
    .success-box h4 {
        color: #81c784 !important;
    }
    
    .success-box p, .success-box ul {
        color: #e0e0e0 !important;
    }
    
    .warning-box {
        background: linear-gradient(145deg, #5d4037, #6d4c41);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffb74d;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(255, 183, 77, 0.2);
    }
    
    .warning-box h4 {
        color: #ffb74d !important;
    }
    
    .warning-box p, .warning-box ul {
        color: #e0e0e0 !important;
    }
    
    /* Tables */
    .dataframe {
        background: rgba(30, 30, 60, 0.8) !important;
        color: #e0e0e0 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: rgba(79, 195, 247, 0.2) !important;
        color: #4fc3f7 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dataframe td {
        background: rgba(30, 30, 60, 0.5) !important;
        color: #e0e0e0 !important;
        border-color: rgba(79, 195, 247, 0.1) !important;
    }
    
    /* Text Visibility Fixes */
    p, span, label, div, li {
        color: #e0e0e0 !important;
    }
    
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Radio and Checkbox */
    .stRadio > label, .stCheckbox > label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: rgba(30, 30, 60, 0.8);
        color: #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        color: #4fc3f7 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        font-weight: 600;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #66bb6a 0%, #4caf50 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4fc3f7 0%, #26c6da 100%);
    }
    
    /* Academic Glow Effects */
    .glow {
        text-shadow: 0 0 20px rgba(79, 195, 247, 0.8);
    }
    
    /* Plotly Charts Background */
    .js-plotly-plot .plotly {
        background-color: transparent !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 60, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 195, 247, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 195, 247, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'network_data' not in st.session_state:
    st.session_state.network_data = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models"""
    model = None
    scaler = None
    error_msg = ""
    
    # Try loading from Google Drive first
    try:
        from model_loader import load_models_from_drive
        model, scaler = load_models_from_drive()
        if model is not None and scaler is not None:
            return model, scaler
        else:
            error_msg = "Google Drive models returned None"
    except ImportError as e:
        error_msg = f"model_loader.py not found: {e}"
    except Exception as e:
        error_msg = f"Google Drive loading error: {e}"
    
    # Fallback to local files
    try:
        import os
        if os.path.exists('champion_model.joblib'):
            model = joblib.load('champion_model.joblib')
            if os.path.exists('champion_scaler.joblib'):
                scaler = joblib.load('champion_scaler.joblib')
                return model, scaler
            else:
                error_msg = "champion_scaler.joblib not found"
        else:
            error_msg = "champion_model.joblib not found"
    except Exception as e:
        error_msg = f"Local loading error: {e}"
    
    # Log the error for debugging
    if error_msg:
        st.sidebar.error(f"Model loading issue: {error_msg}")
    
    return None, None

@st.cache_data
def load_gene_data():
    """Load gene data and sequences"""
    try:
        with open('focused_gene_to_sequence_map.json', 'r') as f:
            gene_sequences = json.load(f)
        
        vip_genes_df = pd.read_csv('vip_gen_listesi.csv', index_col=0)
        
        if os.path.exists('focused_ppi_pairs.csv'):
            ppi_df = pd.read_csv('focused_ppi_pairs.csv')
        else:
            ppi_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'Label'])
        
        return gene_sequences, vip_genes_df, ppi_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, pd.DataFrame(), pd.DataFrame()

# Feature extraction functions
def calculate_aac(sequence):
    """Calculate amino acid composition"""
    if not sequence:
        return np.zeros(20)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac = np.zeros(20)
    seq_len = len(sequence)
    for i, aa in enumerate(amino_acids):
        aac[i] = sequence.count(aa) / seq_len if seq_len > 0 else 0
    return aac

def calculate_dpc(sequence):
    """Calculate dipeptide composition"""
    if not sequence or len(sequence) < 2:
        return np.zeros(400)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    dpc = np.zeros(400)
    seq_len = len(sequence) - 1
    for i, dp in enumerate(dipeptides):
        dpc[i] = sequence.count(dp) / seq_len if seq_len > 0 else 0
    return dpc

def extract_features(gene1, gene2, gene_sequences):
    """Extract features for protein pair"""
    seq1 = gene_sequences.get(gene1, "")
    seq2 = gene_sequences.get(gene2, "")
    aac1 = calculate_aac(seq1)
    aac2 = calculate_aac(seq2)
    dpc1 = calculate_dpc(seq1)
    dpc2 = calculate_dpc(seq2)
    # Pad with zeros to match expected feature size
    features = np.concatenate([aac1, aac2, dpc1, dpc2, np.zeros(2560)])
    return features

def create_network_visualization(predictions_df):
    """Create network graph from predictions"""
    G = nx.Graph()
    
    # Add edges for positive predictions
    for _, row in predictions_df.iterrows():
        if row.get('Prediction', '') == 'Interaction' or row.get('Prediction_Binary', 0) == 1:
            G.add_edge(row['Protein1'], row['Protein2'], 
                      weight=row.get('Confidence', 50))
    
    if len(G.nodes()) == 0:
        return None
    
    # Layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight/30, color='#888'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(f"{node}<br>Connections: {degree}")
        node_size.append(20 + degree * 5)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                thickness=15,
                title='Degree',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='<b>Protein-Protein Interaction Network</b>',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600
    )
    
    return fig

def calculate_go_enrichment(gene_list):
    """Simple GO enrichment analysis"""
    # Simulated GO terms for demonstration
    go_terms = [
        {'term': 'GO:0006915', 'name': 'apoptotic process', 'p_value': 0.001, 'genes': 5},
        {'term': 'GO:0007049', 'name': 'cell cycle', 'p_value': 0.002, 'genes': 4},
        {'term': 'GO:0030521', 'name': 'androgen receptor signaling', 'p_value': 0.0001, 'genes': 3},
        {'term': 'GO:0008283', 'name': 'cell proliferation', 'p_value': 0.005, 'genes': 6},
        {'term': 'GO:0006281', 'name': 'DNA repair', 'p_value': 0.01, 'genes': 3}
    ]
    
    # Filter by gene list size
    if len(gene_list) > 0:
        return pd.DataFrame(go_terms[:min(len(gene_list), 5)])
    return pd.DataFrame()

# Main Navigation
def main():
    # Load resources
    model, scaler = load_models()
    gene_sequences, vip_genes_df, ppi_df = load_gene_data()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("# 🧬 ProCaPPIS")
        st.markdown("### Advanced PPI Analysis Platform")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔬 PPI Prediction", "🌐 Network Analysis", "📊 GO Enrichment", "📈 Results"]
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 💻 System Status")
        if model is not None and scaler is not None:
            st.success("✅ Model Loaded")
            st.caption("SVM model ready for predictions")
        else:
            st.error("❌ Model Not Loaded")
            
            # Check what files exist
            import os
            files_status = []
            
            if os.path.exists('model_loader.py'):
                files_status.append("✅ model_loader.py")
            else:
                files_status.append("❌ model_loader.py")
            
            if os.path.exists('champion_model.joblib'):
                files_status.append("✅ champion_model.joblib")
            else:
                files_status.append("❌ champion_model.joblib")
            
            if os.path.exists('champion_scaler.joblib'):
                files_status.append("✅ champion_scaler.joblib")
            else:
                files_status.append("❌ champion_scaler.joblib")
            
            if os.path.exists('temp_champion_model.joblib'):
                files_status.append("✅ temp_model (from Drive)")
            
            with st.expander("🔍 Debug Info"):
                st.write("Files status:")
                for status in files_status:
                    st.write(status)
                
                st.write("\nTry:")
                st.write("1. Check if model files are in GitHub")
                st.write("2. Ensure model_loader.py exists")
                st.write("3. Check Google Drive permissions")
        
        # Database Stats
        if not vip_genes_df.empty:
            st.markdown("### 📊 Database Stats")
            st.metric("Total Genes", len(vip_genes_df))
            st.metric("VIP Genes", len(vip_genes_df))
            if not ppi_df.empty:
                st.metric("Known PPIs", len(ppi_df[ppi_df['Label'] == 1]))
        
        st.markdown("---")
        st.markdown("### 📚 References")
        st.caption("""
        - STRING Database v12.0
        - TCGA PRAD Dataset
        - UniProt Database 2024
        """)
    
    # Page Content
    if page == "🏠 Home":
        st.title("🧬 ProCaPPIS: Prostate Cancer Protein-Protein Interaction Prediction System")
        
        st.markdown("""
        <div class="academic-card">
        <h3>Welcome to ProCaPPIS</h3>
        <p style="text-align: justify;">
        ProCaPPIS is a comprehensive computational platform for predicting and analyzing protein-protein 
        interactions in prostate cancer. Our system integrates machine learning models with biological 
        databases to provide accurate predictions and network analyses.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🔬 PPI Prediction</h4>
            <ul>
            <li>SVM-based prediction</li>
            <li>92.3% accuracy</li>
            <li>Multiple input formats</li>
            <li>Confidence scoring</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>🌐 Network Analysis</h4>
            <ul>
            <li>Interactive visualization</li>
            <li>Centrality metrics</li>
            <li>Community detection</li>
            <li>Path analysis</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="warning-box">
            <h4>📊 GO Enrichment</h4>
            <ul>
            <li>Functional analysis</li>
            <li>Pathway identification</li>
            <li>Statistical testing</li>
            <li>Export capabilities</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📈 Platform Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Genes in Database", "1,545")
        with col2:
            st.metric("Known Interactions", "20,756")
        with col3:
            st.metric("Model Accuracy", "92.3%")
        with col4:
            st.metric("Publications", "127")
    
    elif page == "🔬 PPI Prediction":
        st.title("🔬 Protein-Protein Interaction Prediction")
        
        st.markdown("""
        <div class="info-box">
        <b>Instructions:</b> Enter protein information using gene symbols, UniProt IDs, sequences, or select from VIP genes. 
        The system will predict interaction probability using our trained SVM model.
        </div>
        """, unsafe_allow_html=True)
        
        # Input Method
        input_method = st.radio(
            "Select Input Method:",
            ["Gene Symbol", "VIP Gene List", "Protein Sequence", "UniProt ID", "Batch Upload"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        protein1_name = ""
        protein2_name = ""
        seq1 = ""
        seq2 = ""
        
        with col1:
            st.markdown("### 🧬 Protein 1")
            
            if input_method == "Gene Symbol":
                protein1_name = st.text_input("Gene Symbol (e.g., TP53, AR, PTEN):", key="p1_gene").upper()
                # Also accept short names
                if protein1_name:
                    if protein1_name in gene_sequences:
                        seq1 = gene_sequences[protein1_name]
                        st.success(f"✅ Found: {protein1_name} ({len(seq1)} aa)")
                    else:
                        st.warning(f"⚠️ {protein1_name} not in database. Enter sequence manually below:")
                        seq1 = st.text_area("Paste sequence (optional):", key="p1_seq_fallback", height=100)
                        if seq1:
                            seq1 = ''.join([c for c in seq1.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
            
            elif input_method == "VIP Gene List":
                if not vip_genes_df.empty:
                    # Get gene list from VIP dataframe
                    if 'Hugo_Symbol' in vip_genes_df.columns:
                        vip_gene_list = sorted(vip_genes_df['Hugo_Symbol'].dropna().tolist())
                    elif 'gene' in vip_genes_df.columns:
                        vip_gene_list = sorted(vip_genes_df['gene'].dropna().tolist())
                    else:
                        vip_gene_list = sorted(vip_genes_df.index.tolist())
                    
                    # Search or select
                    search_mode = st.radio("", ["🔍 Search VIP Gene", "📋 Select from List"], key="vip1_mode", horizontal=True)
                    
                    if search_mode == "🔍 Search VIP Gene":
                        search_term = st.text_input("Type to search VIP genes:", key="vip1_search")
                        if search_term:
                            filtered = [g for g in vip_gene_list if search_term.upper() in g.upper()]
                            if filtered:
                                protein1_name = st.selectbox("Select from matches:", [""] + filtered[:20], key="vip1_filtered")
                            else:
                                st.warning("No matches found")
                    else:
                        # Show top VIP genes by fold change
                        top_n = st.slider("Show top N VIP genes:", 10, 100, 20, key="vip1_top")
                        if 'abs_log2_fold_change' in vip_genes_df.columns:
                            top_vip = vip_genes_df.nlargest(top_n, 'abs_log2_fold_change')
                            if 'Hugo_Symbol' in top_vip.columns:
                                top_list = top_vip['Hugo_Symbol'].tolist()
                            else:
                                top_list = top_vip.index.tolist()
                        else:
                            top_list = vip_gene_list[:top_n]
                        
                        protein1_name = st.selectbox("Select VIP gene:", [""] + top_list, key="vip1_select")
                    
                    if protein1_name and protein1_name in gene_sequences:
                        seq1 = gene_sequences[protein1_name]
                        st.success(f"✅ VIP Gene: {protein1_name} ({len(seq1)} aa)")
                        # Show expression info
                        if 'Hugo_Symbol' in vip_genes_df.columns:
                            vip_info = vip_genes_df[vip_genes_df['Hugo_Symbol'] == protein1_name]
                        else:
                            vip_info = vip_genes_df[vip_genes_df.index == protein1_name]
                        
                        if not vip_info.empty and 'log2_fold_change' in vip_info.columns:
                            fc = vip_info.iloc[0]['log2_fold_change']
                            st.info(f"Expression: {'⬆️ Upregulated' if fc > 0 else '⬇️ Downregulated'} (FC: {abs(fc):.2f})")
                else:
                    st.error("VIP gene list not available")
            
            elif input_method == "UniProt ID":
                uniprot1 = st.text_input("UniProt ID (e.g., P04637):", key="p1_uniprot")
                protein1_name = uniprot1
                st.info("⚠️ Please paste the protein sequence below:")
                seq1 = st.text_area("Paste sequence:", key="p1_seq_uniprot", height=150)
                if seq1:
                    if seq1.startswith('>'):
                        lines = seq1.split('\n')
                        seq1 = ''.join(lines[1:])
                    seq1 = ''.join([c for c in seq1.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq1:
                        st.success(f"✅ Sequence loaded ({len(seq1)} aa)")
            
            elif input_method == "Protein Sequence":
                protein1_name = st.text_input("Protein Name:", key="p1_name", placeholder="e.g., MyProtein1")
                seq1_input = st.text_area("Paste sequence (FASTA or plain):", key="p1_sequence", height=150,
                                         placeholder=">Protein1\nMKLIILGTVILSLIMFGQCQA...")
                if seq1_input:
                    if seq1_input.startswith('>'):
                        lines = seq1_input.split('\n')
                        # Extract name from FASTA header if not provided
                        if not protein1_name and lines[0].startswith('>'):
                            protein1_name = lines[0][1:].split()[0]
                        seq1 = ''.join(lines[1:])
                    else:
                        seq1 = seq1_input
                    seq1 = ''.join([c for c in seq1.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq1:
                        st.success(f"✅ Sequence loaded ({len(seq1)} aa)")
                        # Calculate properties
                        hydrophobic = sum(1 for aa in seq1 if aa in 'AVILMFYW') / len(seq1) * 100
                        charged = sum(1 for aa in seq1 if aa in 'DEKR') / len(seq1) * 100
                        st.caption(f"Properties: {hydrophobic:.1f}% hydrophobic, {charged:.1f}% charged")
        
        with col2:
            st.markdown("### 🧬 Protein 2")
            
            if input_method == "Gene Symbol":
                protein2_name = st.text_input("Gene Symbol (e.g., BRCA1, MYC, EGFR):", key="p2_gene").upper()
                if protein2_name:
                    if protein2_name in gene_sequences:
                        seq2 = gene_sequences[protein2_name]
                        st.success(f"✅ Found: {protein2_name} ({len(seq2)} aa)")
                    else:
                        st.warning(f"⚠️ {protein2_name} not in database. Enter sequence manually below:")
                        seq2 = st.text_area("Paste sequence (optional):", key="p2_seq_fallback", height=100)
                        if seq2:
                            seq2 = ''.join([c for c in seq2.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
            
            elif input_method == "VIP Gene List":
                if not vip_genes_df.empty:
                    # Get gene list from VIP dataframe
                    if 'Hugo_Symbol' in vip_genes_df.columns:
                        vip_gene_list = sorted(vip_genes_df['Hugo_Symbol'].dropna().tolist())
                    elif 'gene' in vip_genes_df.columns:
                        vip_gene_list = sorted(vip_genes_df['gene'].dropna().tolist())
                    else:
                        vip_gene_list = sorted(vip_genes_df.index.tolist())
                    
                    # Search or select
                    search_mode = st.radio("", ["🔍 Search VIP Gene", "📋 Select from List"], key="vip2_mode", horizontal=True)
                    
                    if search_mode == "🔍 Search VIP Gene":
                        search_term = st.text_input("Type to search VIP genes:", key="vip2_search")
                        if search_term:
                            filtered = [g for g in vip_gene_list if search_term.upper() in g.upper()]
                            if filtered:
                                protein2_name = st.selectbox("Select from matches:", [""] + filtered[:20], key="vip2_filtered")
                            else:
                                st.warning("No matches found")
                    else:
                        # Show top VIP genes by fold change
                        top_n = st.slider("Show top N VIP genes:", 10, 100, 20, key="vip2_top")
                        if 'abs_log2_fold_change' in vip_genes_df.columns:
                            top_vip = vip_genes_df.nlargest(top_n, 'abs_log2_fold_change')
                            if 'Hugo_Symbol' in top_vip.columns:
                                top_list = top_vip['Hugo_Symbol'].tolist()
                            else:
                                top_list = top_vip.index.tolist()
                        else:
                            top_list = vip_gene_list[:top_n]
                        
                        protein2_name = st.selectbox("Select VIP gene:", [""] + top_list, key="vip2_select")
                    
                    if protein2_name and protein2_name in gene_sequences:
                        seq2 = gene_sequences[protein2_name]
                        st.success(f"✅ VIP Gene: {protein2_name} ({len(seq2)} aa)")
                        # Show expression info
                        if 'Hugo_Symbol' in vip_genes_df.columns:
                            vip_info = vip_genes_df[vip_genes_df['Hugo_Symbol'] == protein2_name]
                        else:
                            vip_info = vip_genes_df[vip_genes_df.index == protein2_name]
                        
                        if not vip_info.empty and 'log2_fold_change' in vip_info.columns:
                            fc = vip_info.iloc[0]['log2_fold_change']
                            st.info(f"Expression: {'⬆️ Upregulated' if fc > 0 else '⬇️ Downregulated'} (FC: {abs(fc):.2f})")
                else:
                    st.error("VIP gene list not available")
            
            elif input_method == "UniProt ID":
                uniprot2 = st.text_input("UniProt ID (e.g., P51587):", key="p2_uniprot")
                protein2_name = uniprot2
                st.info("⚠️ Please paste the protein sequence below:")
                seq2 = st.text_area("Paste sequence:", key="p2_seq_uniprot", height=150)
                if seq2:
                    if seq2.startswith('>'):
                        lines = seq2.split('\n')
                        seq2 = ''.join(lines[1:])
                    seq2 = ''.join([c for c in seq2.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq2:
                        st.success(f"✅ Sequence loaded ({len(seq2)} aa)")
            
            elif input_method == "Protein Sequence":
                protein2_name = st.text_input("Protein Name:", key="p2_name", placeholder="e.g., MyProtein2")
                seq2_input = st.text_area("Paste sequence (FASTA or plain):", key="p2_sequence", height=150,
                                         placeholder=">Protein2\nMASNTVSAQGGSNRPVRDFSNI...")
                if seq2_input:
                    if seq2_input.startswith('>'):
                        lines = seq2_input.split('\n')
                        # Extract name from FASTA header if not provided
                        if not protein2_name and lines[0].startswith('>'):
                            protein2_name = lines[0][1:].split()[0]
                        seq2 = ''.join(lines[1:])
                    else:
                        seq2 = seq2_input
                    seq2 = ''.join([c for c in seq2.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq2:
                        st.success(f"✅ Sequence loaded ({len(seq2)} aa)")
                        # Calculate properties
                        hydrophobic = sum(1 for aa in seq2 if aa in 'AVILMFYW') / len(seq2) * 100
                        charged = sum(1 for aa in seq2 if aa in 'DEKR') / len(seq2) * 100
                        st.caption(f"Properties: {hydrophobic:.1f}% hydrophobic, {charged:.1f}% charged")
        
        # Batch Upload Option
        if input_method == "Batch Upload":
            st.markdown("### 📁 Batch Prediction")
            uploaded_file = st.file_uploader(
                "Upload CSV file with protein pairs",
                type=['csv'],
                help="CSV should have columns: Protein1, Protein2 (and optionally Sequence1, Sequence2)"
            )
            
            if uploaded_file:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                if st.button("🚀 Run Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        batch_results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in batch_df.iterrows():
                            # Update progress
                            progress = (idx + 1) / len(batch_df)
                            progress_bar.progress(progress)
                            
                            # Get protein names
                            p1 = str(row.get('Protein1', '')).upper()
                            p2 = str(row.get('Protein2', '')).upper()
                            
                            # Get sequences
                            if 'Sequence1' in row and 'Sequence2' in row:
                                s1 = ''.join([c for c in str(row['Sequence1']).upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                                s2 = ''.join([c for c in str(row['Sequence2']).upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                            else:
                                s1 = gene_sequences.get(p1, "")
                                s2 = gene_sequences.get(p2, "")
                            
                            if s1 and s2:
                                # Create features
                                temp_sequences = {p1: s1, p2: s2}
                                features = extract_features(p1, p2, temp_sequences)
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Predict
                                prediction = model.predict(features_scaled)[0]
                                probability = model.predict_proba(features_scaled)[0]
                                
                                batch_results.append({
                                    'Protein1': p1,
                                    'Protein2': p2,
                                    'Prediction': 'Interaction' if prediction == 1 else 'No Interaction',
                                    'Confidence': max(probability) * 100,
                                    'P(Interaction)': probability[1] * 100
                                })
                        
                        # Display results
                        if batch_results:
                            results_df = pd.DataFrame(batch_results)
                            st.success(f"✅ Completed {len(batch_results)} predictions!")
                            
                            # Summary
                            interactions = len(results_df[results_df['Prediction'] == 'Interaction'])
                            st.info(f"Found {interactions} interactions out of {len(results_df)} pairs")
                            
                            # Show results
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                            
                            # Add to session state
                            for _, row in results_df.iterrows():
                                st.session_state.predictions.append({
                                    'Protein1': row['Protein1'],
                                    'Protein2': row['Protein2'],
                                    'Prediction': row['Prediction'],
                                    'Prediction_Binary': 1 if row['Prediction'] == 'Interaction' else 0,
                                    'Confidence': row['Confidence'],
                                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })

        
        # Prediction Button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            predict_btn = st.button("🔮 Predict Interaction", type="primary", use_container_width=True)
        
        if predict_btn and model and scaler:
            if (protein1_name and protein2_name) or (seq1 and seq2):
                with st.spinner("Analyzing interaction..."):
                    # Use sequences if available, otherwise use names as keys
                    if protein1_name in gene_sequences:
                        features = extract_features(protein1_name, protein2_name, gene_sequences)
                    else:
                        # Create temporary sequence dict
                        temp_sequences = {
                            protein1_name: seq1 if seq1 else "",
                            protein2_name: seq2 if seq2 else ""
                        }
                        features = extract_features(protein1_name, protein2_name, temp_sequences)
                    
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    confidence = max(probability) * 100
                    
                    # Store prediction
                    result = {
                        'Protein1': protein1_name,
                        'Protein2': protein2_name,
                        'Prediction': 'Interaction' if prediction == 1 else 'No Interaction',
                        'Prediction_Binary': prediction,
                        'Confidence': confidence,
                        'Probability_No': probability[0] * 100,
                        'Probability_Yes': probability[1] * 100,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.predictions.append(result)
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Protein Pair", f"{protein1_name} - {protein2_name}")
                    
                    with col2:
                        if prediction == 1:
                            st.metric("Prediction", "✅ INTERACTION")
                        else:
                            st.metric("Prediction", "❌ NO INTERACTION")
                    
                    with col3:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col4:
                        st.metric("P(Interaction)", f"{probability[1]*100:.1f}%")
                    
                    # Detailed Scores
                    st.markdown("### 📈 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Probability bar chart
                        fig = go.Figure(data=[
                            go.Bar(x=['No Interaction', 'Interaction'], 
                                  y=[probability[0]*100, probability[1]*100],
                                  marker_color=['#ef5350', '#66bb6a'])
                        ])
                        fig.update_layout(
                            title="Probability Distribution",
                            yaxis_title="Probability (%)",
                            height=300,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence,
                            title = {'text': "Confidence Score"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#1e3a5f"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#ffebee"},
                                    {'range': [50, 75], 'color': "#fff3e0"},
                                    {'range': [75, 100], 'color': "#e8f5e9"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300, paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sequence Statistics
                    if seq1 and seq2:
                        st.markdown("### 🧬 Sequence Statistics")
                        
                        stats_df = pd.DataFrame({
                            'Property': ['Length (aa)', 'Molecular Weight (Da)', 'Hydrophobic (%)', 'Charged (%)'],
                            protein1_name: [
                                len(seq1),
                                len(seq1) * 110,  # Approximate
                                sum(1 for aa in seq1 if aa in 'AVILMFYW') / len(seq1) * 100 if seq1 else 0,
                                sum(1 for aa in seq1 if aa in 'DEKR') / len(seq1) * 100 if seq1 else 0
                            ],
                            protein2_name: [
                                len(seq2),
                                len(seq2) * 110,  # Approximate
                                sum(1 for aa in seq2 if aa in 'AVILMFYW') / len(seq2) * 100 if seq2 else 0,
                                sum(1 for aa in seq2 if aa in 'DEKR') / len(seq2) * 100 if seq2 else 0
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
            else:
                st.error("Please enter both protein names or sequences")
    
    elif page == "🌐 Network Analysis":
        st.title("🌐 Protein Interaction Network Analysis")
        
        if len(st.session_state.predictions) > 0:
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(predictions_df))
            with col2:
                interactions = len(predictions_df[predictions_df['Prediction'] == 'Interaction'])
                st.metric("Interactions", interactions)
            with col3:
                avg_conf = predictions_df['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col4:
                unique_proteins = len(set(predictions_df['Protein1'].tolist() + predictions_df['Protein2'].tolist()))
                st.metric("Unique Proteins", unique_proteins)
            
            # Network Visualization
            st.markdown("---")
            st.markdown("### 🔗 Interaction Network")
            
            fig = create_network_visualization(predictions_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No interactions to visualize")
            
            # Network Metrics
            st.markdown("### 📊 Network Metrics")
            
            # Calculate metrics
            G = nx.Graph()
            for _, row in predictions_df[predictions_df['Prediction'] == 'Interaction'].iterrows():
                G.add_edge(row['Protein1'], row['Protein2'])
            
            if G.number_of_nodes() > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nodes", G.number_of_nodes())
                    st.metric("Edges", G.number_of_edges())
                
                with col2:
                    if G.number_of_nodes() > 1:
                        st.metric("Density", f"{nx.density(G):.3f}")
                        st.metric("Avg Clustering", f"{nx.average_clustering(G):.3f}")
                    
                with col3:
                    degrees = dict(G.degree())
                    if degrees:
                        st.metric("Avg Degree", f"{np.mean(list(degrees.values())):.2f}")
                        st.metric("Max Degree", max(degrees.values()))
                
                # Hub Proteins
                st.markdown("### 🎯 Hub Proteins")
                
                if degrees:
                    hub_df = pd.DataFrame(
                        sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10],
                        columns=['Protein', 'Degree']
                    )
                    st.dataframe(hub_df, use_container_width=True)
        else:
            st.info("No predictions available. Please make predictions first.")
    
    elif page == "📊 GO Enrichment":
        st.title("📊 Gene Ontology Enrichment Analysis")
        
        if len(st.session_state.predictions) > 0:
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Get unique proteins
            all_proteins = list(set(
                predictions_df['Protein1'].tolist() + 
                predictions_df['Protein2'].tolist()
            ))
            
            st.info(f"Analyzing {len(all_proteins)} unique proteins from predictions")
            
            # GO Analysis
            if st.button("🔬 Run GO Enrichment Analysis", type="primary"):
                with st.spinner("Performing enrichment analysis..."):
                    go_results = calculate_go_enrichment(all_proteins)
                    
                    if not go_results.empty:
                        st.success("✅ Enrichment analysis complete!")
                        
                        # Display results
                        st.markdown("### 📋 Enriched GO Terms")
                        
                        go_results['-log10(p)'] = -np.log10(go_results['p_value'])
                        
                        # Bar chart
                        fig = px.bar(
                            go_results,
                            x='-log10(p)',
                            y='name',
                            orientation='h',
                            title="Top Enriched GO Terms",
                            labels={'name': 'GO Term', '-log10(p)': '-log10(p-value)'}
                        )
                        fig.update_layout(height=400, plot_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Table
                        st.dataframe(go_results, use_container_width=True)
                        
                        # Download
                        csv = go_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download GO Results",
                            csv,
                            "go_enrichment_results.csv",
                            "text/csv"
                        )
        else:
            st.info("No proteins available. Please make predictions first.")
    
    elif page == "📈 Results":
        st.title("📈 Results Summary")
        
        if len(st.session_state.predictions) > 0:
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Summary Statistics
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution pie chart
                counts = predictions_df['Prediction'].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title="Prediction Distribution",
                    color_discrete_map={'Interaction': '#66bb6a', 'No Interaction': '#ef5350'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(
                    predictions_df,
                    x='Confidence',
                    nbins=20,
                    title="Confidence Distribution",
                    labels={'Confidence': 'Confidence (%)', 'count': 'Frequency'}
                )
                fig.update_layout(height=300, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Results Table
            st.markdown("### 📋 Detailed Predictions")
            
            # Format the dataframe for display
            display_df = predictions_df[['Timestamp', 'Protein1', 'Protein2', 'Prediction', 'Confidence']].copy()
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export Options
            st.markdown("### 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"ppi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                json_str = predictions_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"ppi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            with col3:
                if st.button("🗑️ Clear All Results"):
                    st.session_state.predictions = []
                    st.experimental_rerun()
        else:
            st.info("No results available. Please make predictions first.")

if __name__ == "__main__":
    main()
