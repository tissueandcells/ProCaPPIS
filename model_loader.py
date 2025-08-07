import os
import gdown
import streamlit as st
import joblib

@st.cache_resource
def load_models_from_drive():
    """Load model from Google Drive and scaler from GitHub"""
    
    MODEL_FILE_ID = '1VgRc_mQwhiADsulKAQ-GeTRy3Jn6yLRf'
    
    model = None
    scaler = None
    
    # Model dosya adı
    model_path = "champion_model.joblib"  # temp_ olmadan
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from Google Drive...'):
            try:
                url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
                gdown.download(url, model_path, quiet=False)
                st.success('Model downloaded successfully')
            except Exception as e:
                st.error(f'Failed to download model: {e}')
                return None, None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        return None, None
    
    # Scaler'ı yükle
    if os.path.exists('champion_scaler.joblib'):
        try:
            scaler = joblib.load('champion_scaler.joblib')
        except Exception as e:
            st.error(f'Failed to load scaler: {e}')
            return None, None
    else:
        st.error("champion_scaler.joblib not found")
        return None, None
    
    return model, scaler
