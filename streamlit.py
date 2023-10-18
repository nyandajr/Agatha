# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

@st.cache_resource()
def load_model():
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model('R:/Agatha/my_maize_disease_model.h5', custom_objects=custom_objects)
    return model

model = load_model()

# Define a function to make predictions on the image
def predict_image(img, model):
    # Resize and preprocess the image for the model
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Convert to float32 in [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Model expects batches of images

    # Get predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Here, replace this list with the order of classes as used during training
    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    return class_names[predicted_class]

# Streamlit App UI
st.title('Maize Disease Prediction App')

st.write("""
Upload an image of maize, and this app will predict if it has a disease.
""")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        prediction = predict_image(img, model)
        st.write(f"Prediction: {prediction}")

    except Exception as e:
        st.write(f"An error occurred: {e}")
        st.write("Please upload a valid image with one of the following extensions: jpg, jpeg, png.")
