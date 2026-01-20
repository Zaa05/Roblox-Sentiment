import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
from datetime import datetime

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Analisis Sentimen Roblox",
    page_icon="🎮",
    layout="wide"
)

# ========== CSS STYLING ==========
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive { background-color: #d4edda; border-left: 5px solid #28a745; }
    .neutral { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .negative { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<h1 class="main-title">🎮 Analisis Sentimen Pengguna Roblox</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Menggunakan Model BernoulliNB dengan TF-IDF</h3>', unsafe_allow_html=True)

# ========== FUNGSI LOAD MODEL ==========
@st.cache_resource
def load_models():
    """Load semua model dan komponen"""
    try:
        # Load model BernoulliNB
        model = joblib.load('models/bernoulli_nb_model.pkl')
        
        # Load TF-IDF vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load feature selector
        selector = joblib.load('models/feature_selector.pkl')
        
        # Load label mapping
        with open('models/label_mapping.json', 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # Reverse mapping untuk konversi angka ke label
        label_to_sentiment = {v: k for k, v in label_mapping.items()}
        
        return model, vectorizer, selector, label_mapping, label_to_sentiment
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Pastikan semua file model ada di folder 'models/'")
        return None, None, None, None, None

# ========== FUNGSI PREPROCESSING ==========
def preprocess_text(text):
    """Preprocessing teks untuk prediksi"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove emoji
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stopwords untuk Bahasa Indonesia
    stopwords = set([
        'yang', 'dan', 'di', 'dari', 'dalam', 'dengan', 'untuk', 'pada', 'ke', 'para',
        'oleh', 'karena', 'itu', 'ini', 'atau', 'juga', 'tidak', 'bukan', 'saja', 'akan',
        'ada', 'sudah', 'harus', 'lebih', 'bisa', 'dapat', 'jika', 'agar', 'bahwa', 'sebagai',
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
        'game', 'roblox', 'main', 'robloxnya', 'gamenya', 'seru', 'bagus', 'jelek', 'lag', 'lemot'
    ])
    
    # Tokenization dan remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords and len(word) > 2]
    
    return ' '.join(tokens)

# ========== LOAD MODEL ==========
model, vectorizer, selector, label_mapping, label_to_sentiment = load_models()

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://cdn.worldvectorlogo.com/logos/roblox-1.svg", width=150)
    
    st.markdown("### 📊 Informasi Model")
    st.info("""
    **Model:** Bernoulli Naive Bayes  
    **Feature:** TF-IDF  
    **Kelas:** 3 Sentimen  
    
    **Label Mapping:**  
    • 0 = Negatif  
    • 1 = Netral  
    • 2 = Positif
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Contoh Komentar")
    
    example_comments = [
        "Game Roblox ini sangat seru dan menghibur!",
        "Grafis biasa saja, tidak istimewa tapi bisa dimainkan",
        "Sangat buruk, sering lag dan bug dimana-mana"
    ]
    
    for i, comment in enumerate(example_comments):
        if st.button(f"Contoh {i+1}", key=f"example_{i}"):
            st.session_state.input_text = comment
            st.rerun()

# ========== TAB UTAMA ==========
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Sentimen", "📊 Batch Processing", "ℹ️  Informasi"])

with tab1:
    # Input Text Area
    st.markdown("### ✍️ Masukkan Komentar tentang Roblox")
    
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    input_text = st.text_area(
        "Komentar:",
        value=st.session_state.input_text,
        height=150,
        placeholder="Contoh: Game Roblox ini sangat seru, grafisnya bagus banget!",
        key="input_text_area"
    )
    
    # Tombol Aksi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        predict_btn = st.button("🚀 Prediksi Sentimen", type="primary", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)
    
    with col3:
        example_btn = st.button("📋 Contoh", use_container_width=True)
    
    # Handle tombol
    if clear_btn:
        st.session_state.input_text = ""
        if 'prediction_result' in st.session_state:
            del st.session_state.prediction_result
        st.rerun()
    
    if example_btn:
        st.session_state.input_text = "Game Roblox ini sangat seru, grafisnya bagus banget!"
        st.rerun()
    
    # PREDIKSI
    if predict_btn and input_text.strip():
        if model is None:
            st.error("⚠️ Model belum dimuat. Periksa file model.")
        else:
            with st.spinner("🔍 Menganalisis sentimen..."):
                try:
                    # Preprocess
                    cleaned_text = preprocess_text(input_text)
                    
                    # Transform dengan TF-IDF
                    text_tfidf = vectorizer.transform([cleaned_text])
                    
                    # Feature selection
                    text_selected = selector.transform(text_tfidf)
                    
                    # Binerisasi untuk BernoulliNB
                    text_binary = (text_selected > 0).astype(int)
                    
                    # Prediksi
                    prediction = model.predict(text_binary)[0]
                    
                    # Konversi ke label sentimen
                    sentiment = label_to_sentiment[prediction]
                    
                    # Get probabilities
                    probabilities = model.predict_proba(text_binary)[0]
                    
                    # Simpan hasil ke session state
                    st.session_state.prediction_result = {
                        'text': input_text,
                        'cleaned_text': cleaned_text,
                        'sentiment': sentiment,
                        'prediction': prediction,
                        'probabilities': probabilities.tolist(),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {str(e)}")
    
    # TAMPILKAN HASIL
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")
        
        # Warna berdasarkan sentimen
        sentiment_colors = {
            'positif': ('#28a745', '✅'),
            'netral': ('#ffc107', '⚖️'),
            'negatif': ('#dc3545', '❌')
        }
        
        color, icon = sentiment_colors[result['sentiment']]
        
        # Tampilkan hasil utama
        st.markdown(f"""
        <div style='
            background-color: {color}20;
            border-left: 5px solid {color};
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        '>
            <h3 style='color: {color}; margin: 0;'>
                {icon} <strong>{result['sentiment'].upper()}</strong>
            </h3>
            <p style='margin: 5px 0;'>Kode: {result['prediction']}</p>
            <p style='margin: 5px 0;'>Waktu: {result['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detail teks
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📝 Teks Original", expanded=True):
                st.write(result['text'])
        
        with col2:
            with st.expander("🧹 Teks Setelah Cleaning", expanded=False):
                st.write(result['cleaned_text'] if result['cleaned_text'] else "(kosong)")
        
        # Probabilitas
        st.markdown("#### 📈 Probabilitas Sentimen")
        
        prob_df = pd.DataFrame({
            'Sentimen': ['Negatif', 'Netral', 'Positif'],
            'Probabilitas': result['probabilities'],
            'Persentase': [f"{p*100:.1f}%" for p in result['probabilities']]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            st.bar_chart(prob_df.set_index('Sentimen')['Probabilitas'])
        
        with col2:
            # Tabel
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # DOWNLOAD HASIL
        st.markdown("---")
        st.markdown("### 💾 Download Hasil")
        
        # Buat DataFrame untuk download
        result_df = pd.DataFrame([{
            'Komentar': result['text'],
            'Komentar_Cleaned': result['cleaned_text'],
            'Sentimen': result['sentiment'],
            'Kode_Sentimen': result['prediction'],
            'Prob_Negatif': f"{result['probabilities'][0]:.4f}",
            'Prob_Netral': f"{result['probabilities'][1]:.4f}",
            'Prob_Positif': f"{result['probabilities'][2]:.4f}",
            'Waktu_Analisis': result['timestamp']
        }])
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sentimen_roblox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON
            json_data = json.dumps(result, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"sentimen_roblox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

with tab2:
    st.markdown("### 📊 Analisis Batch (Multiple Komentar)")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom 'komentar'",
        type=['csv'],
        help="File harus memiliki kolom bernama 'komentar'"
    )
    
    if uploaded_file is not None:
        try:
            # Load file
            df_upload = pd.read_csv(uploaded_file)
            
            if 'komentar' not in df_upload.columns:
                st.error("❌ File harus memiliki kolom 'komentar'")
            else:
                st.success(f"✅ File berhasil diupload! ({len(df_upload)} baris)")
                
                # Preview data
                with st.expander("👁️ Preview Data Upload"):
                    st.dataframe(df_upload.head(), use_container_width=True)
                
                # Tombol analisis
                if st.button("🔍 Analisis Semua Komentar", type="primary", use_container_width=True):
                    with st.spinner(f"🔄 Menganalisis {len(df_upload)} komentar..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in enumerate(df_upload.itertuples(), 1):
                            try:
                                # Preprocess
                                cleaned_text = preprocess_text(row.komentar)
                                
                                # Transform
                                text_tfidf = vectorizer.transform([cleaned_text])
                                text_selected = selector.transform(text_tfidf)
                                text_binary = (text_selected > 0).astype(int)
                                
                                # Prediksi
                                prediction = model.predict(text_binary)[0]
                                sentiment = label_to_sentiment[prediction]
                                
                                # Probabilitas
                                probabilities = model.predict_proba(text_binary)[0]
                                
                                results.append({
                                    'No': idx,
                                    'Komentar': row.komentar,
                                    'Komentar_Cleaned': cleaned_text,
                                    'Sentimen': sentiment,
                                    'Kode_Sentimen': prediction,
                                    'Prob_Negatif': probabilities[0],
                                    'Prob_Netral': probabilities[1],
                                    'Prob_Positif': probabilities[2]
                                })
                                
                            except Exception as e:
                                results.append({
                                    'No': idx,
                                    'Komentar': row.komentar,
                                    'Komentar_Cleaned': 'ERROR',
                                    'Sentimen': 'ERROR',
                                    'Kode_Sentimen': -1,
                                    'Prob_Negatif': 0,
                                    'Prob_Netral': 0,
                                    'Prob_Positif': 0
                                })
                            
                            # Update progress bar
                            progress_bar.progress(idx / len(df_upload))
                        
                        # Buat DataFrame hasil
                        results_df = pd.DataFrame(results)
                        
                        # Hapus progress bar
                        progress_bar.empty()
                        
                        # Tampilkan hasil
                        st.markdown("####  Hasil Analisis")
                        
                        # Statistik
                        st.markdown("#####  Statistik Sentimen")
                        sentiment_counts = results_df['Sentimen'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Tabel statistik
                            stats_df = pd.DataFrame({
                                'Sentimen': sentiment_counts.index,
                                'Jumlah': sentiment_counts.values,
                                'Persentase': (sentiment_counts.values / len(results_df) * 100).round(1)
                            })
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Pie chart
                            st.bar_chart(sentiment_counts)
                        
                        # Preview hasil
                        with st.expander("👁️ Preview Hasil (10 baris pertama)"):
                            st.dataframe(results_df.head(10), use_container_width=True)
                        
                        # DOWNLOAD HASIL BATCH
                        st.markdown("---")
                        st.markdown("### 💾 Download Hasil Batch")
                        
                        csv_batch = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label=f"📥 Download {len(results_df)} Hasil (CSV)",
                            data=csv_batch,
                            file_name=f"hasil_batch_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab3:
    st.markdown("### ℹ️  Informasi Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ####  Tentang Aplikasi
        Aplikasi ini digunakan untuk menganalisis sentimen pengguna 
        terhadap game **Roblox** menggunakan model **Machine Learning**.
        
        ####  Teknologi
        - **Framework**: Streamlit
        - **Model**: Bernoulli Naive Bayes
        - **Feature**: TF-IDF Vectorization
        - **Bahasa**: Python 3.8+
        
        ####  Fitur
        - Prediksi sentimen tunggal
        - Analisis batch file CSV
        - Download hasil (CSV/JSON)
        - Visualisasi probabilitas
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Spesifikasi Model
        - **Algoritma**: Bernoulli Naive Bayes
        - **Fitur**: TF-IDF dengan seleksi fitur
        - **Kelas**: 3 (Negatif, Netral, Positif)
        - **Bahasa Input**: Indonesia
        
        #### ⚠️  Catatan
        - Model dilatih dengan data Bahasa Indonesia
        - Hasil untuk bahasa lain mungkin kurang akurat
        - Model khusus untuk analisis sentimen Roblox
        
        #### 🔧 Untuk Pengembang
        - Source code tersedia
        - Mudah dikustomisasi
        - Support batch processing
        """)
    
    # Model info
    st.markdown("---")
    st.markdown("####  Informasi Model yang Dimuat")
    
    if model is not None:
        model_info = {
            'Model Type': type(model).__name__,
            'Number of Features': text_binary.shape[1] if 'text_binary' in locals() else 'N/A',
            'Classes': list(label_to_sentiment.values()),
            'Label Mapping': label_mapping
        }
        
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(model_info.items())[:2]:
                st.info(f"**{key}**: {value}")
        with col2:
            for key, value in list(model_info.items())[2:]:
                st.info(f"**{key}**: {value}")
    else:
        st.warning("Model belum dimuat. Periksa file di folder 'models/'")

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>© 2026 | Analisis Sentimen Roblox | TF-IDF + BernoulliNB | Skripsi</p>
        <p style='font-size: 0.9em;'>
            Dibangun dengan ❤️ menggunakan Streamlit & Scikit-learn
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
