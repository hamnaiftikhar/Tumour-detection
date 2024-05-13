# -*- coding: utf-8 -*-
"""
Created on Mon May  6 01:08:23 2024

@author: hamna
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained model
model = tf.keras.models.load_model("model/my_model.h5")

# Define labels
labels = ["Tumor", "No Tumor"]

# Set custom colors
primary_color = "#3498db"  # Blue
secondary_color = "#e74c3c"  # Red
text_color = "#FAF8ED"  # Dark gray
background_color = "#99256E"  # Light gray

# Configure Streamlit settings
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def predict(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("Brain Tumor Detection")
    st.markdown("---")
    st.markdown("Upload an MRI image of a brain to detect if a tumor is present.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.subheader("Uploaded MRI Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Make prediction
        prediction = predict(uploaded_file, model)

        st.subheader("Prediction")
        result = labels[np.argmax(prediction)]

        st.write(f"The model predicts: **{result}**")

        # Display additional information based on prediction
        if result == "No Tumor":
            st.success("No tumor detected. You're all clear!")
        else:
            st.warning("A tumor is detected. Please consult a medical professional.")

            # Adjusting colors for the presence of a tumor
            st.markdown(f'<style>div.stButton > button {{background-color: {secondary_color} !important}}</style>',
                        unsafe_allow_html=True)



if __name__ == "__main__":
    main()
