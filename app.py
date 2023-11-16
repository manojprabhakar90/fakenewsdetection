# app.py

import streamlit as st
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.saved_model.load('saved_model.pb')
    print("TensorFlow Model Loaded")
    return model

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def predict(model, text):
    processed_text = preprocess_text(text)
    prediction = model.signatures["serving_default"](tf.constant([processed_text]))
    pred_value = prediction['output'][0].numpy()
    return "Real" if pred_value > 0.5 else "Fake"
model = load_model()
st.title("Fake News Detection")
news_text = st.text_area("Enter News Text", "")

if st.button("Predict"):
    prediction = predict(model, news_text)
    st.write("Prediction:", prediction)
