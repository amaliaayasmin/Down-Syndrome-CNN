import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Load the pre-trained model
MODEL_PATH = "downSynd_model.h5"
model = load_model(MODEL_PATH)

# App title
st.title("Down Syndrome Prediction from Image")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("This application uses a pre-trained model to predict whether an image depicts Down Syndrome or a healthy individual. Upload an image to get started.")

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the target size expected by the model
    target_size = (250, 250)  # Updated to match the model's expected input size
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Image upload section
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(processed_image)
        result = "Down Syndrome" if prediction[0][0] > 0.5 else "Healthy"
        confidence = prediction[0][0] if result == "Down Syndrome" else 1 - prediction[0][0]

        # Display prediction result
        st.subheader("Prediction")
        st.write(f"Result: {result}")
        st.write(f"Confidence: {confidence:.2f}")

# Footer
st.sidebar.info("Developed using Streamlit and TensorFlow")
