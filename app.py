# app.py

import streamlit as st
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Function to load the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.saved_model.load('./saved_model.pb')
    return model

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Reconstruct the text
    return ' '.join(filtered_words)

# Function to make predictions
def predict(model, text):
    processed_text = preprocess_text(text)

    # Predict (modify according to your model's signature)
    prediction = model.signatures["serving_default"](tf.constant([processed_text]))
    
    # Extracting prediction value (modify as per your model's output)
    pred_value = prediction['output'][0].numpy()

    return "Real" if pred_value > 0.5 else "Fake"

# Load the model
model = load_model()

# Streamlit User Interface
st.title("Fake News Detection")
news_text = st.text_area("Enter News Text", "")

if st.button("Predict"):
    prediction = predict(model, news_text)
    st.write("Prediction:", prediction)
