# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow_hub as hub

MODEL_DRIVE_ID = 'YOUR_MODEL_FILE_ID'  # Replace with your actual model's Google Drive ID
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'

@st.cache_resource()
def load_model():
    # Download and load the model from Google Drive
    gdown.download(MODEL_URL, 'my_maize_disease_model.h5', quiet=False)
    
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model('my_maize_disease_model.h5', custom_objects=custom_objects)
    
    return model

model = load_model()

# Define a function to make predictions on the image
def predict_image(img, model):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    return class_names[predicted_class]

# Streamlit UI
st.title('Maize Disease Prediction App')

st.write("""
Upload an image of maize, and this app will predict if it has a disease.
""")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        prediction = predict_image(img, model)
        st.write(f"Prediction: {prediction}")
    except Exception as e:
        st.write(f"Error: {e}")
