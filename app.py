import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Filter App", layout="centered")

st.title("🖼️ Image Processing Web App")
st.markdown("Upload an image and apply various filters.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Available filters
filters = [
    "Grayscale",
    "Canny Edge Detection",
    "Gaussian Blur",
    "Thresholding",
    "Sepia",
    "Negative",
    "Cartoon Effect"
]

# Show dropdown only if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    img_np = np.array(image.convert('RGB'))  # Convert PIL image to numpy array (RGB)

    selected_filter = st.selectbox("Select a filter to apply", filters)

    if st.button("Apply Filter"):
        result = None

        if selected_filter == "Grayscale":
            result = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        elif selected_filter == "Canny Edge Detection":
            result = cv2.Canny(img_np, 100, 200)

        elif selected_filter == "Gaussian Blur":
            result = cv2.GaussianBlur(img_np, (15, 15), 0)

        elif selected_filter == "Thresholding":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        elif selected_filter == "Sepia":
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            result = cv2.transform(img_np, sepia_filter)
            result = np.clip(result, 0, 255).astype(np.uint8)

        elif selected_filter == "Negative":
            result = cv2.bitwise_not(img_np)

        elif selected_filter == "Cartoon Effect":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img_np, 9, 300, 300)
            result = cv2.bitwise_and(color, color, mask=edges)

        st.markdown("### Filtered Image:")
        st.image(result, use_column_width=True, clamp=True)
