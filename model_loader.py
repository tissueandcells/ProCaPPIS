import os
import gdown
import streamlit as st
import joblib

@st.cache_resource
def load_models_from_drive():
    """Load model from Google Drive and scaler from GitHub with improved error handling"""
    
    MODEL_FILE_ID = '1VgRc_mQwhiADsulKAQ-GeTRy3Jn6yLRf'
    
    model = None
    scaler = None
    
    # Model dosya yolu
    model_path = "champion_model.joblib"
    
    # 1. Model'i Google Drive'dan indir
    st.write("🔄 Checking model availability...")
    
    # Eğer model dosyası yoksa indir
    if not os.path.exists(model_path):
        st.write("📥 Downloading model from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
            gdown.download(url, model_path, quiet=False)
            
            # İndirme başarılı mı kontrol et
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:  # En az 1KB olmalı
                st.success('✅ Model downloaded successfully')
            else:
                st.error('❌ Model download failed - file is empty or corrupted')
                return None, None
                
        except Exception as e:
            st.error(f'❌ Failed to download model: {str(e)}')
            st.write("**Possible solutions:**")
            st.write("- Check internet connection")
            st.write("- Verify Google Drive file ID")
            st.write("- Try refreshing the page")
            return None, None
    else:
        st.write("✅ Model file already exists")
    
    # 2. Model'i yükle
    try:
        st.write("🔄 Loading model...")
        model = joblib.load(model_path)
        st.success(f'✅ Model loaded: {type(model).__name__}')
    except Exception as e:
        st.error(f'❌ Failed to load model: {str(e)}')
        # Bozuk dosyayı sil ki tekrar indirilsin
        try:
            os.remove(model_path)
            st.write("🗑️ Corrupted model file removed, please refresh to re-download")
        except:
            pass
        return None, None
    
    # 3. Scaler'ı yükle - farklı dosya adlarını dene
    scaler_files = [
        'champion_scaler.joblib',
        'best_scaler.joblib', 
        'scaler.joblib',
        'standard_scaler.joblib'
    ]
    
    scaler_loaded = False
    for scaler_file in scaler_files:
        if os.path.exists(scaler_file):
            try:
                st.write(f"🔄 Loading scaler from {scaler_file}...")
                scaler = joblib.load(scaler_file)
                st.success(f'✅ Scaler loaded: {type(scaler).__name__}')
                scaler_loaded = True
                break
            except Exception as e:
                st.warning(f'⚠️ Failed to load {scaler_file}: {str(e)}')
                continue
    
    if not scaler_loaded:
        st.error("❌ No scaler file found or loadable")
        st.write("**Looking for files:**")
        for scaler_file in scaler_files:
            status = "✅ Found" if os.path.exists(scaler_file) else "❌ Missing"
            st.write(f"- {scaler_file}: {status}")
        return None, None
    
    # 4. Model ve scaler uyumluluğunu kontrol et
    try:
        # Test features (840 dimensional - AAC(40) + DPC(800))
        import numpy as np
        test_features = np.random.randn(1, 840)
        
        # Scaler test
        scaled_features = scaler.transform(test_features)
        
        # Model test
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)
        
        st.success("✅ Model and scaler compatibility verified")
        st.write(f"📊 Expected input features: {test_features.shape[1]}")
        st.write(f"📊 Model prediction shape: {prediction.shape}")
        st.write(f"📊 Probability shape: {probability.shape}")
        
    except Exception as e:
        st.error(f"❌ Model-scaler compatibility issue: {str(e)}")
        st.write("**This might indicate:**")
        st.write("- Feature dimension mismatch")
        st.write("- Incompatible model/scaler versions") 
        st.write("- Corrupted files")
        return None, None
    
    return model, scaler


def check_system_requirements():
    """Check if all required packages are available"""
    required_packages = ['gdown', 'joblib', 'streamlit', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        st.write("Install with:")
        st.code(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def get_file_info():
    """Get information about model files"""
    files_info = []
    
    files_to_check = [
        'champion_model.joblib',
        'champion_scaler.joblib', 
        'best_model.joblib',
        'best_scaler.joblib',
        'model.joblib',
        'scaler.joblib'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_mb = size / (1024 * 1024)
            files_info.append({
                'file': file,
                'exists': True,
                'size_mb': f"{size_mb:.2f} MB",
                'size_bytes': size
            })
        else:
            files_info.append({
                'file': file,
                'exists': False,
                'size_mb': "N/A",
                'size_bytes': 0
            })
    
    return files_info


if __name__ == "__main__":
    st.title("🧪 Model Loader Test")
    
    # Requirements check
    if not check_system_requirements():
        st.stop()
    
    # File info
    st.write("### 📁 File Information")
    files_info = get_file_info()
    
    for info in files_info:
        status = "✅" if info['exists'] else "❌"
        st.write(f"{status} {info['file']} - {info['size_mb']}")
    
    # Test loading
    if st.button("🧪 Test Load Models"):
        model, scaler = load_models_from_drive()
        
        if model is not None and scaler is not None:
            st.success("🎉 Both model and scaler loaded successfully!")
        else:
            st.error("❌ Failed to load models")
