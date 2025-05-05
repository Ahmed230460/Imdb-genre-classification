import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import scipy.sparse
import os

# Set page config
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load artifacts with error handling
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        # Load small files normally
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            artifacts['tfidf'] = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        with open('director_encoder.pkl', 'rb') as f:
            artifacts['director_encoder'] = pickle.load(f)
        with open('content4_encoder.pkl', 'rb') as f:
            artifacts['content4_encoder'] = pickle.load(f)
        with open('genre_encoder.pkl', 'rb') as f:
            artifacts['genre_encoder'] = pickle.load(f)
        
        # Handle large model file
        if os.path.exists('rf_model.zip'):
            with zipfile.ZipFile('rf_model.zip', 'r') as zip_ref:
                with zip_ref.open('rf_model.pkl') as f:
                    artifacts['model'] = pickle.load(f)
        elif os.path.exists('rf_model.pkl'):
            with open('rf_model.pkl', 'rb') as f:
                artifacts['model'] = pickle.load(f)
        else:
            st.error("Model file not found")
            return None
            
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None

# Text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) 
              for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def main():
    st.title("🎬 Movie Genre Classifier")
    st.markdown("Predict a movie's genre based on its synopsis and metadata")
    
    # Load artifacts with progress indicator
    with st.spinner("Loading models..."):
        artifacts = load_artifacts()
    
    if not artifacts:
        st.error("Failed to load required models. Please check the deployment.")
        return
    
    # Input form
    with st.form("movie_input"):
        st.subheader("Enter Movie Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            synopsis = st.text_area("Synopsis", height=200,
                                  placeholder="Enter the movie plot summary...",
                                  help="At least 50 characters works best")
            director = st.text_input("Director", placeholder="e.g., Christopher Nolan")
            content4 = st.text_input("Main Actor/Actress", placeholder="e.g., Leonardo DiCaprio")
            
        with col2:
            rating = st.slider("Rating (IMDb)", 1.0, 10.0, 7.0, 0.1,
                             help="Average IMDb rating from 1-10")
            votes = st.number_input("Number of Votes", min_value=0, value=1000,
                                  help="Approximate number of IMDb votes")
        
        submitted = st.form_submit_button("Predict Genre")
    
    if submitted:
        if not synopsis or len(synopsis) < 20:
            st.error("Please enter a valid movie synopsis (at least 20 characters)")
            return
            
        try:
            with st.spinner("Analyzing..."):
                # Preprocess inputs
                processed_text = preprocess_text(synopsis)
                text_features = artifacts['tfidf'].transform([processed_text])
                
                # Encode categorical features with fallbacks
                director = director.strip().lower() if director else "unknown"
                content4 = content4.strip().lower() if content4 else "unknown"
                
                try:
                    director_encoded = artifacts['director_encoder'].transform([director])[0]
                except:
                    director_encoded = 0  # Fallback for unknown directors
                    
                try:
                    content4_encoded = artifacts['content4_encoder'].transform([content4])[0]
                except:
                    content4_encoded = 0  # Fallback for unknown actors
                
                # Prepare numeric features
                numeric_features = np.array([[rating, votes, director_encoded, content4_encoded]])
                numeric_features_scaled = artifacts['scaler'].transform(numeric_features)
                
                # Combine features
                features = scipy.sparse.hstack([text_features, numeric_features_scaled])
                
                # Make prediction
                prediction = artifacts['model'].predict(features)
                predicted_genre = artifacts['genre_encoder'].inverse_transform(prediction)[0]
                
                # Show results
                st.success(f"Predicted Genre: **{predicted_genre}**")
                
                # Show prediction probabilities
                with st.expander("View detailed genre probabilities"):
                    proba = artifacts['model'].predict_proba(features)[0]
                    genres = artifacts['genre_encoder'].classes_
                    proba_df = pd.DataFrame({
                        "Genre": genres,
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)
                    
                    st.dataframe(
                        proba_df.style.format({"Probability": "{:.2%}"}),
                        height=min(400, 35 * len(proba_df)),
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please try different input values")

if __name__ == "__main__":
    main()
