import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

# Load pre-trained models and encoders
@st.cache_resource
def load_artifacts():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/director_encoder.pkl', 'rb') as f:
        director_encoder = pickle.load(f)
    with open('models/content4_encoder.pkl', 'rb') as f:
        content4_encoder = pickle.load(f)
    with open('models/genre_encoder.pkl', 'rb') as f:
        genre_encoder = pickle.load(f)
    with open('models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, scaler, director_encoder, content4_encoder, genre_encoder, model

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Main app function
def main():
    st.title("🎬 Movie Genre Classifier")
    st.markdown("Predict a movie's genre based on its synopsis and metadata")
    
    # Load artifacts
    tfidf, scaler, director_encoder, content4_encoder, genre_encoder, model = load_artifacts()
    
    # Input form
    with st.form("movie_input"):
        st.subheader("Enter Movie Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            synopsis = st.text_area("Synopsis", height=200,
                                  placeholder="Enter the movie plot summary...")
            director = st.text_input("Director", placeholder="Director name")
            content4 = st.text_input("Main Actor/Actress", placeholder="Primary actor name")
            
        with col2:
            rating = st.slider("Rating (IMDb)", 1.0, 10.0, 5.0, 0.1)
            votes = st.number_input("Number of Votes", min_value=0, value=1000)
        
        submitted = st.form_submit_button("Predict Genre")
    
    if submitted:
        if not synopsis:
            st.error("Please enter a movie synopsis")
            return
            
        try:
            # Preprocess inputs
            processed_text = preprocess_text(synopsis)
            text_features = tfidf.transform([processed_text])
            
            # Encode categorical features
            director_encoded = director_encoder.transform([director.lower().strip()])[0] if director else 0
            content4_encoded = content4_encoder.transform([content4.lower().strip()])[0] if content4 else 0
            
            # Prepare numeric features
            numeric_features = np.array([[rating, votes, director_encoded, content4_encoded]])
            numeric_features_scaled = scaler.transform(numeric_features)
            
            # Combine features
            features = scipy.sparse.hstack([text_features, numeric_features_scaled])
            
            # Make prediction
            prediction = model.predict(features)
            predicted_genre = genre_encoder.inverse_transform(prediction)[0]
            
            # Show results
            st.success(f"Predicted Genre: **{predicted_genre}**")
            
            # Show prediction probabilities
            st.subheader("Genre Probabilities")
            proba = model.predict_proba(features)[0]
            genres = genre_encoder.classes_
            proba_df = pd.DataFrame({
                "Genre": genres,
                "Probability": proba
            }).sort_values("Probability", ascending=False)
            
            st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), height=400)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
