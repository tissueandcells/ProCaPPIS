# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import h5py
import requests
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import itertools
from datetime import datetime
import time
import warnings
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ProCaPPIS - Advanced PPI Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN (CSS) ---
# Function to embed a local background image
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_background(local_img_path):
    bin_str = get_base64_of_bin_file(local_img_path)
    if bin_str:
        img_format = local_img_path.split('.')[-1]
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(10, 20, 40, 0.8), rgba(10, 20, 40, 0.8)), url("data:image/{img_format};base64,{bin_str}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Try to set a background image, otherwise the gradient will be used.
# set_background('your_background_image.png')

# Advanced, modern dark theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto+Slab:wght@700&display=swap');

    /* Default background if no image is found */
    .stApp {
        background-image: linear-gradient(to bottom right, #0b1a2e, #1d2a4a, #2a3963);
        background-attachment: fixed;
        background-size: cover;
        color: #e0e0e0; /* Default text color */
    }

    /* Main content area */
    .main {
        padding: 2rem;
        background-color: transparent;
    }

    /* "Frosted glass" effect cards */
    .academic-card {
        background: rgba(15, 26, 46, 0.7); /* Dark semi-transparent */
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }

    /* Header styles */
    h1, h2 {
        color: #ffffff !important;
        font-family: 'Roboto Slab', serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    h1 {
        border-bottom: 3px solid #00aaff;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    h3 {
        color: #cceeff !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }

    /* Custom info boxes */
    .info-box, .success-box, .warning-box, .error-box {
        padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
        font-family: 'Montserrat', sans-serif; color: #ffffff;
        background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(5px);
    }
    .info-box { border-left: 5px solid #00aaff; }
    .success-box { border-left: 5px solid #00ff88; }
    .warning-box { border-left: 5px solid #ffaa00; }
    .error-box { border-left: 5px solid #ff4444; }

    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #00aaff 0%, #0077cc 100%);
        color: white; border: none; padding: 0.75rem 2.5rem; font-weight: 600;
        border-radius: 8px; font-family: 'Montserrat', sans-serif;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 170, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 170, 255, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 26, 46, 0.8);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Input fields for dark theme */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>textarea, .stMultiSelect>div>div>div>div {
        background-color: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        color: #ffffff !important;
    }
    .stSelectbox>div>div>div, .stMultiSelect>div>div>div>div {
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] li {
        background-color: #1d2a4a; color: #e0e0e0;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #2a3963;
    }

    /* Metric boxes */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1); padding: 1.5rem;
        border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);
    }
    [data-testid="metric-container"] [data-testid="metric-label"] { color: #a0b0c0; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: #ffffff; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1); color: #a0b0c0;
        border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #00aaff; color: white; border-color: #00aaff;
    }
    </style>
    """, unsafe_allow_html=True)


# --- DATA LOADING AND PROCESSING FUNCTIONS ---

@st.cache_resource
def load_all_data():
    """Loads and caches all necessary data files."""
    data = {}
    try:
        # Load all required files
        data['sequences'] = json.load(open('focused_gene_to_sequence_map.json', 'r'))
        data['embeddings'] = h5py.File('focused_esm_embeddings.h5', 'r')
        data['ppi_pairs'] = pd.read_csv('focused_ppi_pairs.csv')
        data['vip_genes'] = pd.read_csv('vip_gen_listesi.csv')
        data['model'] = joblib.load('champion_model.joblib')
        data['scaler'] = joblib.load('champion_scaler.joblib')
        return data
    except Exception as e:
        st.markdown(f'<div class="error-box">Data or model loading error: {e}. Please ensure all required files (`.joblib`, `.csv`, `.json`, `.h5`) are in the same directory as the app.</div>', unsafe_allow_html=True)
        return None

# --- FEATURE EXTRACTION ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]

def extract_features(seq1, seq2, gene1, gene2, data):
    """Extracts features for a protein pair."""
    # Amino Acid Composition (AAC)
    aac1 = np.zeros(20)
    if seq1:
        for i, acid in enumerate(AMINO_ACIDS): aac1[i] = seq1.count(acid) / len(seq1)
    
    aac2 = np.zeros(20)
    if seq2:
        for i, acid in enumerate(AMINO_ACIDS): aac2[i] = seq2.count(acid) / len(seq2)

    # Dipeptide Composition (DPC)
    dpc1 = np.zeros(400)
    if seq1 and len(seq1) > 1:
        for i, dip in enumerate(DIPEPTIDES): dpc1[i] = seq1.count(dip) / (len(seq1) - 1)

    dpc2 = np.zeros(400)
    if seq2 and len(seq2) > 1:
        for i, dip in enumerate(DIPEPTIDES): dpc2[i] = seq2.count(dip) / (len(seq2) - 1)

    seq_features = np.concatenate([aac1, aac2, dpc1, dpc2])
    
    # ESM Embeddings
    emb1, emb2 = np.zeros(1280), np.zeros(1280)
    if data and gene1 and gene2 and 'embeddings' in data:
        try:
            if gene1 in data['embeddings']: emb1 = data['embeddings'][gene1][:]
            if gene2 in data['embeddings']: emb2 = data['embeddings'][gene2][:]
        except Exception:
            pass # Silently fail if embeddings are not found
    
    esm_features = np.concatenate([emb1, emb2])
    return np.concatenate([seq_features, esm_features])

# --- PAGE FUNCTIONS ---

def home_page():
    st.markdown("# 🧬 ProCaPPIS: Prostate Cancer PPI Prediction System")
    st.markdown("""
        <div class="academic-card">
            <h3 style="text-align: center;">Advanced Bioinformatics Analysis Platform</h3>
            <p style="text-align: justify; font-size: 16px; line-height: 1.8;">
            Welcome to ProCaPPIS. This platform utilizes advanced machine learning algorithms, trained on comprehensive data from the STRING database and TCGA expression profiles, to predict protein-protein interactions specific to prostate cancer.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="academic-card"><h4>🔬 Scientific Rigor</h4><ul style="line-height: 1.8;"><li>Support Vector Machine (SVM)</li><li>92.3% Prediction Accuracy</li><li>Validated on 20,756 interaction pairs</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="academic-card"><h4>📊 Advanced Analytics</h4><ul style="line-height: 1.8;"><li>Interactive Network Visualization</li><li>Statistical Significance Testing</li><li>Confidence Score Calculation</li></ul></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="academic-card"><h4>🎓 Academic Tools</h4><ul style="line-height: 1.8;"><li>Batch Prediction Capability</li><li>Result Export (CSV)</li><li>GO Enrichment Analysis</li></ul></div>', unsafe_allow_html=True)

def page_ppi_prediction(data):
    st.markdown("<h1>🔬 Protein-Protein Interaction Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("<h3>Input Configuration</h3>", unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose input method:",
        ["Database Search", "Manual Sequence Entry"],
        horizontal=True,
        key="ppi_input_method"
    )
    
    seq1_final, seq2_final = "", ""
    gene1, gene2 = "", ""
    use_embeddings = (input_method == "Database Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Protein 1</h4>", unsafe_allow_html=True)
        if input_method == "Database Search":
            available_genes = sorted(list(data['sequences'].keys()))
            gene1 = st.selectbox("Select gene:", [""] + available_genes, key="gene1_select")
            if gene1: seq1_final = data['sequences'].get(gene1, "")
        else: # Manual Entry
            gene1 = st.text_input("Protein Name/ID", placeholder="e.g., MyProtein1", key="gene1_manual")
            seq1_input = st.text_area("Protein Sequence (FASTA or plain)", height=100, key="seq1_manual_area")
            if seq1_input:
                lines = seq1_input.strip().split('\n')
                seq1_final = ''.join(lines[1:]) if lines[0].startswith('>') else ''.join(lines)
                seq1_final = ''.join([c.upper() for c in seq1_final if c.upper() in AMINO_ACIDS])

    with col2:
        st.markdown("<h4>Protein 2</h4>", unsafe_allow_html=True)
        if input_method == "Database Search":
            available_genes = sorted(list(data['sequences'].keys()))
            gene2 = st.selectbox("Select gene:", [""] + available_genes, key="gene2_select")
            if gene2: seq2_final = data['sequences'].get(gene2, "")
        else: # Manual Entry
            gene2 = st.text_input("Protein Name/ID", placeholder="e.g., MyProtein2", key="gene2_manual")
            seq2_input = st.text_area("Protein Sequence (FASTA or plain)", height=100, key="seq2_manual_area")
            if seq2_input:
                lines = seq2_input.strip().split('\n')
                seq2_final = ''.join(lines[1:]) if lines[0].startswith('>') else ''.join(lines)
                seq2_final = ''.join([c.upper() for c in seq2_final if c.upper() in AMINO_ACIDS])

    if st.button("🚀 Predict Interaction", type="primary", use_container_width=True, disabled=(not seq1_final or not seq2_final)):
        with st.spinner("Analyzing protein interaction..."):
            features = extract_features(seq1_final, seq2_final, gene1, gene2, data)
            
            features_scaled = data['scaler'].transform(features.reshape(1, -1))
            prediction = data['model'].predict(features_scaled)[0]
            probability = data['model'].predict_proba(features_scaled)[0]
            confidence = max(probability) * 100

            st.session_state.last_prediction = {
                "gene1": gene1 or "Manual1", "gene2": gene2 or "Manual2",
                "prediction": prediction, "confidence": confidence
            }
    
    st.markdown('</div>', unsafe_allow_html=True) # Close academic-card

    if 'last_prediction' in st.session_state:
        res = st.session_state.last_prediction
        st.markdown("---")
        st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        pred_text = "INTERACTION ✅" if res['prediction'] == 1 else "NO INTERACTION ❌"
        c1.metric(f"Result for {res['gene1']} - {res['gene2']}", pred_text)
        c2.metric("Confidence Score", f"{res['confidence']:.2f}%")
        st.progress(int(res['confidence']))
        st.markdown('</div>', unsafe_allow_html=True)

def page_network_analysis(data):
    st.markdown("<h1>🕸️ Protein Interaction Network Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("<h3>Network Configuration</h3>", unsafe_allow_html=True)
    
    vip_genes_list = data['vip_genes'].index.tolist()
    default_selection = vip_genes_list[:15] if len(vip_genes_list) > 15 else vip_genes_list
    
    genes_to_analyze = st.multiselect(
        "Select genes from the VIP list to build the network:",
        options=vip_genes_list,
        default=default_selection
    )

    if st.button("🔍 Create Network", type="primary", use_container_width=True):
        if genes_to_analyze:
            with st.spinner("Creating network..."):
                G = nx.Graph()
                positive_ppi = data['ppi_pairs'][data['ppi_pairs']['Label'] == 1]
                relevant_edges = positive_ppi[
                    (positive_ppi['Gene1'].isin(genes_to_analyze)) & 
                    (positive_ppi['Gene2'].isin(genes_to_analyze))
                ]
                for _, row in relevant_edges.iterrows():
                    G.add_edge(row['Gene1'], row['Gene2'])
                for gene in genes_to_analyze:
                    if gene not in G: G.add_node(gene)
                
                st.session_state['current_network'] = G
                st.markdown(f'<div class="success-box">Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Please select at least one gene to create a network.</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True) # Close academic-card

    if 'current_network' in st.session_state and st.session_state['current_network'] is not None:
        G = st.session_state['current_network']
        st.markdown("---")
        st.markdown("<h2>Network Visualization & Metrics</h2>", unsafe_allow_html=True)
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=0.8, iterations=50)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color='rgba(255,255,255,0.5)'), hoverinfo='none', mode='lines')
            
            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                degree = G.degree(node)
                node_x.append(x); node_y.append(y)
                node_text.append(f'{node}<br>Degree: {degree}')
                node_color.append(degree)
                node_size.append(10 + degree * 2)

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[str(node) for node in G.nodes()], textposition="top center",
                                    hoverinfo='text', hovertext=node_text, textfont=dict(size=10, color="#ffffff"),
                                    marker=dict(showscale=True, colorscale='Blues', size=node_size, color=node_color,
                                                colorbar=dict(thickness=15, title='Node Degree', xanchor='left', titleside='right'),
                                                line=dict(width=2, color='rgba(0,170,255,0.8)')))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(showlegend=False, hovermode='closest',
                                            margin=dict(b=0,l=0,r=0,t=0),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="info-box">The network is empty.</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

def page_go_enrichment(data):
    st.markdown("<h1>🧬 Gene Ontology (GO) Enrichment Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("<h3>Analysis Configuration</h3>", unsafe_allow_html=True)
    
    vip_genes_list = data['vip_genes'].index.tolist()
    default_selection = vip_genes_list[:20] if len(vip_genes_list) > 20 else vip_genes_list
    
    gene_list = st.multiselect(
        "Enter or select gene symbols for enrichment analysis:",
        options=vip_genes_list,
        default=default_selection
    )
    p_threshold = st.number_input("P-value (FDR) threshold:", 0.001, 1.0, 0.05, 0.01)

    if st.button("🔬 Perform Enrichment Analysis", type="primary", use_container_width=True):
        if len(gene_list) >= 3:
            with st.spinner("Querying STRING database for enrichment..."):
                string_api_url = "https://string-db.org/api/json/enrichment"
                params = {'identifiers': '%0d'.join(gene_list), 'species': 9606, 'caller_identity': 'ProCaPPIS_App'}
                try:
                    response = requests.post(string_api_url, data=params)
                    if response.status_code == 200 and response.json():
                        results_df = pd.DataFrame(response.json())
                        significant = results_df[results_df['fdr'] <= p_threshold]
                        st.markdown(f'<div class="success-box">Found {len(significant)} significant GO terms.</div>', unsafe_allow_html=True)
                        st.session_state.go_results = significant
                    else:
                        st.markdown('<div class="error-box">Could not retrieve results from STRING. Check gene symbols or API status.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">API request failed: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Please select at least 3 genes for analysis.</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True) # Close academic-card

    if 'go_results' in st.session_state and not st.session_state.go_results.empty:
        results = st.session_state.go_results
        st.markdown("---")
        st.markdown("<h2>Enrichment Results</h2>", unsafe_allow_html=True)
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        
        results['-log10(FDR)'] = -np.log10(results['fdr'].clip(lower=1e-50))
        top_results = results.head(20)
        
        fig = px.bar(top_results, x='-log10(FDR)', y='description', orientation='h',
                     title="Top 20 Enriched GO Terms", color='category',
                     color_discrete_map={'Process': '#00aaff', 'Function': '#00ff88', 'Component': '#ffaa00'},
                     labels={'description': 'GO Term', '-log10(FDR)': '-log10(FDR)'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Full Data Table"):
            st.dataframe(results)
            
        st.markdown('</div>', unsafe_allow_html=True)

def page_relevance_check(data):
    st.markdown("<h1>🔍 Prostate Cancer Relevance Checker</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("<h3>Check Protein Relevance</h3>", unsafe_allow_html=True)
    
    vip_df = data['vip_genes']
    protein_input = st.text_input("Enter a protein/gene symbol to check its relevance:", "AR")

    if st.button("🔍 Check Relevance", type="primary", use_container_width=True):
        if protein_input:
            protein_upper = protein_input.strip().upper()
            if protein_upper in vip_df.index:
                details = vip_df.loc[protein_upper]
                score = (abs(details['log2_fold_change']) / vip_df['abs_log2_fold_change'].max()) * 100
                
                st.markdown(f'<div class="success-box">**{protein_upper}** is a VIP gene in our prostate cancer dataset.</div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Relevance Score", f"{score:.1f}%")
                c2.metric("Log2 Fold Change", f"{details['log2_fold_change']:.3f}", "Upregulated" if details['log2_fold_change'] > 0 else "Downregulated")
                
                st.progress(int(score))
                with st.expander("View Detailed Expression Data"):
                    st.json(details.to_dict())
            else:
                st.markdown(f'<div class="warning-box">**{protein_upper}** was not found in our VIP gene list for prostate cancer.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">Please enter a protein symbol.</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True) # Close academic-card

# --- MAIN APP STRUCTURE ---
def main():
    # Load data once and pass it to pages
    data = load_all_data()
    if data is None:
        st.markdown('<div class="error-box">Application cannot start without data. Please check file paths and try again.</div>', unsafe_allow_html=True)
        return

    with st.sidebar:
        st.markdown("# 🧬 ProCaPPIS")
        st.markdown("---")
        
        st.markdown("## 📍 Navigation")
        pages = {
            "🏠 Home": "home",
            "🔬 PPI Prediction": "prediction",
            "🕸️ Network Analysis": "network",
            "🧬 GO Enrichment": "go_enrichment",
            "🔍 Relevance Check": "relevance_check"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
        
        st.markdown("---")
        st.markdown("## 💻 System Status")
        st.markdown('<div class="success-box">✅ Models & Data Loaded</div>', unsafe_allow_html=True)
        
        st.markdown("## 📊 Database Stats")
        st.metric("Total Genes", len(data['sequences']))
        st.metric("Total Interactions", len(data['ppi_pairs']))
        
        st.markdown("---")
        st.info("Version: 2.1.0 | Academic Use License")

    # Page routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "prediction":
        page_ppi_prediction(data)
    elif st.session_state.page == "network":
        page_network_analysis(data)
    elif st.session_state.page == "go_enrichment":
        page_go_enrichment(data)
    elif st.session_state.page == "relevance_check":
        page_relevance_check(data)

if __name__ == "__main__":
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = {}
    if 'current_network' not in st.session_state:
        st.session_state.current_network = None
    if 'go_results' not in st.session_state:
        st.session_state.go_results = pd.DataFrame()
        
    main()
