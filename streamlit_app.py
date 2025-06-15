import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Load model dan vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("Klasifikasi Sentimen Komentar Spotify")

user_input = st.text_area("Masukkan komentar pengguna:")

if st.button("Prediksi Sentimen"):
    if user_input:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "Positive":
            st.success("Hasil Prediksi: Sentimen Positif 😊")
        else:
            st.error("Hasil Prediksi: Sentimen Negatif 😞")
    else:
        st.warning("Silakan masukkan komentar terlebih dahulu.")
