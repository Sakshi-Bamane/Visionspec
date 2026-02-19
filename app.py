import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from src.gradcam import apply_gradcam

st.set_page_config(page_title="VisionSpec QC", layout="centered")

st.title("🔍 VisionSpec QC – PCB Defect Detection")
st.write(
    "Upload a PCB image. The system predicts **PASS / DEFECT** "
    "and highlights defect regions using **Grad-CAM**."
)

# load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/mobilenet_qc.h5")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload PCB Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # preprocess for prediction
    resized = cv2.resize(image, (224, 224))
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # predict
    prediction = model.predict(input_img)[0][0]
    label = "DEFECT ❌" if prediction > 0.5 else "PASS ✅"

    st.subheader(f"Prediction: {label}")

    # save temp image
    os.makedirs("outputs", exist_ok=True)
    temp_path = "outputs/temp_input.jpg"
    cv2.imwrite(temp_path, image)

    # apply gradcam
    gradcam_img = apply_gradcam(temp_path, model)

    # display images
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_column_width=True
    )

    st.image(
        cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB),
        caption="Grad-CAM Visualization",
        use_column_width=True
    )

    # cleanup
    os.remove(temp_path)
