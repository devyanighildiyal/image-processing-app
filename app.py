import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Filter App", layout="centered")

# Centered title
st.markdown("<h1 style='text-align: center;'>🖼️ Image Processing Filter</h1>", unsafe_allow_html=True)
st.markdown("Upload an image and apply various filters.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

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

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))

    st.image(image, caption="Original Image", use_container_width=True)

    selected_filter = st.selectbox("Select a filter to apply", filters)

    # Additional inputs for certain filters
    if selected_filter == "Resize Image":
        new_width = st.number_input("New Width", min_value=10, max_value=1000, value=img_np.shape[1])
        new_height = st.number_input("New Height", min_value=10, max_value=1000, value=img_np.shape[0])
    elif selected_filter == "Rotate Image":
        angle = st.slider("Rotation Angle", -180, 180, 0)

    if st.button("Apply Filter"):
        result = None

        if selected_filter == "Convert to Grayscale":
            result = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        elif selected_filter == "Resize Image":
            result = cv2.resize(img_np, (int(new_width), int(new_height)))

        elif selected_filter == "Rotate Image":
            (h, w) = img_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img_np, M, (w, h))

        elif selected_filter == "Apply Gaussian Blur":
            result = cv2.GaussianBlur(img_np, (15, 15), 0)

        elif selected_filter == "Apply Median Blur":
            result = cv2.medianBlur(img_np, 5)

        elif selected_filter == "Canny Edge Detection":
            result = cv2.Canny(img_np, 100, 200)

        elif selected_filter == "Apply Thresholding":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        elif selected_filter == "Detect Contours":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = img_np.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        elif selected_filter == "Show Color Histogram":
            fig, ax = plt.subplots()
            colors = ('r', 'g', 'b')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
            st.pyplot(fig)

        # Show image only if result exists
        if result is not None:
            st.markdown("### Filtered Image:")
            st.image(result, use_container_width=True, clamp=True)
