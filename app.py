import streamlit as st
import cv2
import numpy as np
from collections import Counter
import pytesseract
import tempfile
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Feature Extraction Functions ---
def extract_color_features(image, k=3):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = Counter(labels.flatten())
    freqs = np.array(list(counts.values())) / sum(counts.values())
    color_diversity = -np.sum(freqs * np.log(freqs + 1e-6))
    return color_diversity

def extract_text_features_tesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    sizes = []
    highlighted_image = image.copy()
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50 and data['text'][i].strip() != "":
            x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
            sizes.append(w * h)
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    avg_text_area = np.mean(sizes) if sizes else 0
    text_density = len(sizes) / (image.shape[0] * image.shape[1]) if sizes else 0
    return avg_text_area, text_density, highlighted_image

def extract_whitespace_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = image.shape[0] * image.shape[1]
    return white_pixels / total_pixels

def extract_edge_complexity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_complexity = np.sum(edges) / (image.shape[0] * image.shape[1])
    return edge_complexity, edges

def extract_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return brightness

def extract_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return saturation

def extract_features(image):
    color_diversity = extract_color_features(image)
    avg_text_area, text_density, highlighted_text_image = extract_text_features_tesseract(image)
    whitespace_ratio = extract_whitespace_ratio(image)
    edge_complexity, edge_image = extract_edge_complexity(image)
    brightness = extract_brightness(image)
    saturation = extract_saturation(image)
    return {
        "Color Diversity": color_diversity,
        "Avg Text Area": avg_text_area,
        "Text Density": text_density,
        "Whitespace Ratio": whitespace_ratio,
        "Edge Complexity": edge_complexity,
        "Brightness": brightness,
        "Saturation": saturation,
        "highlighted_text_image": highlighted_text_image,
        "edge_image": edge_image
    }

# --- Streamlit UI ---
st.title("KV Analyzer (CV-based)")

uploaded_file = st.file_uploader("Upload an image file for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    analyze_btn = st.button("Analyze")
    if analyze_btn:
        st.info("Analyzing image... Please wait.")
        progress = st.progress(0)
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        image = cv2.imread(tmp_path)
        progress.progress(20)
        features = extract_features(image)
        progress.progress(80)
        st.success("Analysis complete!")
        progress.progress(100)

        # Show visualizations
        st.subheader("Highlighted Text Regions")
        st.image(cv2.cvtColor(features["highlighted_text_image"], cv2.COLOR_BGR2RGB), use_column_width=True)
        st.subheader("Edge Complexity Visualization")
        st.image(features["edge_image"], use_column_width=True, channels="GRAY")

        # Show feature values and descriptions
        parameter_descriptions = {
            "Color Diversity": "Measures the variety of colors in the image. Higher values indicate a more diverse color palette.",
            "Avg Text Area": "Represents the average size of text regions detected in the image. Larger values suggest bigger text areas.",
            "Text Density": "Indicates the proportion of text regions relative to the total image area. Higher values suggest text-heavy images.",
            "Whitespace Ratio": "Measures the proportion of white or blank space in the image. Higher values indicate more whitespace.",
            "Edge Complexity": "Quantifies the density of edges in the image, representing visual complexity.",
            "Brightness": "Represents the average brightness level of the image. Higher values indicate brighter images.",
            "Saturation": "Measures the intensity of colors in the image. Higher values suggest more vibrant and saturated colors."
        }
        st.subheader("Feature Analysis")
        for name in ["Color Diversity", "Avg Text Area", "Text Density", "Whitespace Ratio", "Edge Complexity", "Brightness", "Saturation"]:
            value = features[name]
            description = parameter_descriptions[name]
            formatted_value = f"{value:.9f}" if value < 0.01 else f"{value:.4f}"
            st.write(f"**{name}:** {formatted_value}  \n_{description}_")

        # Highlight Color Diversity and Edge Complexity
        color_diversity = features["Color Diversity"]
        edge_complexity = features["Edge Complexity"]
        st.subheader("Key Insights")
        great_color = color_diversity > 0.9
        great_edge = edge_complexity < 9
        if great_color and great_edge:
            st.success("✅ This image has great color diversity and edge complexity!")
        else:
            if not great_color:
                st.warning("⚠️ Color diversity is below the recommended threshold (0.9). Consider using a more diverse color palette.")
            if not great_edge:
                st.warning("⚠️ Edge complexity is above the recommended threshold (9). Consider simplifying the visual complexity.")

        st.markdown("---")