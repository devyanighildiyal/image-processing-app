import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Processing Filter", layout="centered")

# Centered title using markdown and HTML
st.markdown("<h1 style='text-align: center;'>🖼️ Image Processing Filter</h1>", unsafe_allow_html=True)
st.markdown("Upload an image and apply various filters and transformations.")

# Upload image
uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png"])

# New filter list
filters = [
    "Convert to Grayscale",
    "Resize Image",
    "Rotate Image",
    "Apply Gaussian Blur",
    "Apply Median Blur",
    "Canny Edge Detection",
    "Apply Thresholding",
    "Detect Contours",
    "Show Color Histogram"
]

# Proceed if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    img_np = np.array(image.convert('RGB'))

    selected_filter = st.selectbox("🎨 Select a filter to apply", filters)

    if st.button("Apply Filter"):
        result = None

        if selected_filter == "Convert to Grayscale":
            result = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            st.image(result, caption="Grayscale Image", use_container_width=True, clamp=True)

        elif selected_filter == "Resize Image":
            width = st.slider("Select new width", 50, 800, img_np.shape[1])
            height = st.slider("Select new height", 50, 800, img_np.shape[0])
            result = cv2.resize(img_np, (width, height))
            st.image(result, caption="Resized Image", use_container_width=True)

        elif selected_filter == "Rotate Image":
            angle = st.slider("Rotation Angle", -180, 180, 90)
            (h, w) = img_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img_np, M, (w, h))
            st.image(result, caption=f"Rotated Image ({angle}°)", use_container_width=True)

        elif selected_filter == "Apply Gaussian Blur":
            result = cv2.GaussianBlur(img_np, (15, 15), 0)
            st.image(result, caption="Gaussian Blurred Image", use_container_width=True)

        elif selected_filter == "Apply Median Blur":
            result = cv2.medianBlur(img_np, 5)
            st.image(result, caption="Median Blurred Image", use_container_width=True)

        elif selected_filter == "Canny Edge Detection":
            result = cv2.Canny(img_np, 100, 200)
            st.image(result, caption="Canny Edge Detection", use_container_width=True, clamp=True)

        elif selected_filter == "Apply Thresholding":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            st.image(result, caption="Thresholded Image", use_container_width=True, clamp=True)

        elif selected_filter == "Detect Contours":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edged = cv2.Canny(gray, 30, 150)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contoured_img = img_np.copy()
            cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)
            st.image(contoured_img, caption="Image with Contours", use_container_width=True)

        elif selected_filter == "Show Color Histogram":
            color = ('r', 'g', 'b')
            fig, ax = plt.subplots()
            for i, col in enumerate(color):
                hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
                ax.set_xlim([0, 256])
            ax.set_title("Color Histogram")
            st.pyplot(fig)
