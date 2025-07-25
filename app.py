import streamlit as st
import cv2
import numpy as np
from collections import Counter
import pytesseract
import tempfile
import matplotlib.pyplot as plt
import pickle
import json
from scipy.optimize import minimize

# --- Load Model and Dependencies ---
@st.cache_resource
def load_model_files():
    try:
        with open('models/kv_reach_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('models/feature_ranges.pkl', 'rb') as f:
            feature_ranges = pickle.load(f)
            
        with open('models/correlations.pkl', 'rb') as f:
            correlations = pickle.load(f)
            
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return model, feature_ranges, correlations, metadata
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

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

# --- Model Functions ---
def find_optimal_features(target_reach, model, feature_ranges):
    """Find optimal color diversity and saturation values to achieve target reach"""
    def objective(features):
        color_diversity, saturation = features
        predicted_reach = model.predict([[color_diversity, saturation]])[0]
        return (predicted_reach - target_reach) ** 2
    
    # Initial guess (middle of the range)
    x0 = [
        (feature_ranges['color_diversity']['min'] + feature_ranges['color_diversity']['max']) / 2,
        (feature_ranges['saturation']['min'] + feature_ranges['saturation']['max']) / 2
    ]
    
    # Define bounds for optimization
    bounds = [
        (feature_ranges['color_diversity']['min'], feature_ranges['color_diversity']['max']),
        (feature_ranges['saturation']['min'], feature_ranges['saturation']['max'])
    ]
    
    # Run optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    # Get optimal values
    optimal_color_diversity, optimal_saturation = result.x
    
    # Verify predicted reach with these values
    predicted_reach = model.predict([[optimal_color_diversity, optimal_saturation]])[0]
    
    return {
        'color_diversity': optimal_color_diversity,
        'saturation': optimal_saturation,
        'predicted_reach': predicted_reach,
        'target_reach': target_reach,
        'error': abs(predicted_reach - target_reach)
    }

# --- Streamlit UI ---
st.title("KV Analyzer (CV-based) v3.0")

# Load the model and related data
model, feature_ranges, correlations, metadata = load_model_files()
model_loaded = model is not None

# Add target reach input
st.sidebar.header("Target Reach Settings")
if model_loaded:
    # Calculate min/max possible reach based on model
    min_possible_reach = model.predict([ [
        feature_ranges['color_diversity']['min' if correlations[0] > 0 else 'max'],
        feature_ranges['saturation']['min' if correlations[1] > 0 else 'max']
    ]])[0]
    
    max_possible_reach = model.predict([ [
        feature_ranges['color_diversity']['max' if correlations[0] > 0 else 'min'],
        feature_ranges['saturation']['max' if correlations[1] > 0 else 'min']
    ]])[0]
    
    # Round to nearest 1000 for better UX
    min_reach = round(min_possible_reach / 1000) * 1000
    max_reach = round(max_possible_reach / 1000) * 1000
    default_reach = round((min_reach + max_reach) / 2 / 1000) * 1000
    
    st.sidebar.info(f"Based on our model, we can predict reaches between {int(min_reach)} and {int(max_reach)}")
    target_reach = st.sidebar.number_input("Expected Total Reach", 
                                         min_value=int(min_reach * 0.9), 
                                         max_value=int(max_reach * 1.1),
                                         value=int(default_reach),
                                         step=500)
    
    # Calculate optimal feature values for this target
    if model_loaded:
        optimal = find_optimal_features(target_reach, model, feature_ranges)
        st.sidebar.subheader("Recommended Values")
        st.sidebar.write(f"For a reach of {int(target_reach)}:")
        st.sidebar.write(f"- Color Diversity: {optimal['color_diversity']:.4f}")
        st.sidebar.write(f"- Saturation: {optimal['saturation']:.4f}")
        st.sidebar.write(f"(Expected reach: {int(optimal['predicted_reach'])}, Error: {optimal['error']:.2f})")
else:
    st.sidebar.warning("Model not loaded. Can't calculate target reach recommendations.")
    target_reach = st.sidebar.number_input("Expected Total Reach", value=10000, step=1000)

uploaded_file = st.file_uploader("Upload an image file for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
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
        st.image(cv2.cvtColor(features["highlighted_text_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.subheader("Edge Complexity Visualization")
        st.image(features["edge_image"], use_container_width=True, channels="GRAY")

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

        # Compare image features with optimal values
        if model_loaded:
            color_diversity = features["Color Diversity"]
            saturation = features["Saturation"]
            
            # If we have a target reach, show comparison with optimal values
            if 'optimal' in locals():
                st.subheader(f"Reach Analysis for Target: {int(target_reach)}")
                
                # Calculate predicted reach for the current image
                current_reach = model.predict([[color_diversity, saturation]])[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    # Change delta calculation and coloring:
                    # Now, when color_diversity < optimal, it shows DOWN arrow in GREEN
                    # When color_diversity > optimal, it shows UP arrow in RED
                    delta = color_diversity - optimal['color_diversity']  # Direct calculation (no reversal)
                    st.metric("Current Color Diversity", f"{color_diversity:.4f}", 
                             f"{delta:.4f}", delta_color="inverse")  # inverse makes negative values green
                with col2:
                    st.metric("Maximum Threshold", f"{optimal['color_diversity']:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Same change for saturation
                    delta = saturation - optimal['saturation']  # Direct calculation
                    st.metric("Current Saturation", f"{saturation:.4f}", 
                             f"{delta:.4f}", delta_color="inverse")  # inverse makes negative values green
                with col2:
                    st.metric("Maximum Threshold", f"{optimal['saturation']:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Show the difference between current and target reach
                    reach_diff = current_reach - target_reach
                    st.metric("Predicted Reach", f"{int(current_reach)}", 
                             f"{int(reach_diff)}")
                with col2:
                    st.metric("Target Reach", f"{int(target_reach)}")
                
                # New logic: check if predicted reach exceeds target reach
                if current_reach >= target_reach:
                    st.success("✅ This image is predicted to exceed your target reach!")
                else:
                    st.warning("⚠️ This image may not reach your target audience size.")
                    
                    recommendations = []
                    if correlations[0] < 0:  # Negative correlation for color diversity
                        if color_diversity > optimal['color_diversity']:
                            recommendations.append("- Reduce color diversity by using fewer colors in your image")
                        else:
                            recommendations.append("- You can increase color diversity slightly")
                    else:
                        if color_diversity < optimal['color_diversity']:
                            recommendations.append("- Increase color diversity by using more colors in your image")
                        else:
                            recommendations.append("- You can reduce color diversity slightly")
                            
                    if correlations[1] < 0:  # Negative correlation for saturation
                        if saturation > optimal['saturation']:
                            recommendations.append("- Reduce saturation by using more muted, less vibrant colors")
                        else:
                            recommendations.append("- You can increase saturation slightly")
                    else:
                        if saturation < optimal['saturation']:
                            recommendations.append("- Increase saturation by using more vibrant colors")
                        else:
                            recommendations.append("- You can reduce saturation slightly")
                    
                    st.subheader("Recommendations:")
                    for rec in recommendations:
                        st.write(rec)

        # # Basic insights (legacy code)
        # st.subheader("General Insights")
        # great_color = color_diversity < 1.0037
        # great_saturation = saturation < 35.6761
        # if great_color and great_saturation:
        #     st.success("✅ This image has great color diversity and saturation according to general benchmarks!")
        # else:
        #     if not great_color:
        #         st.warning("⚠️ Color diversity is above the recommended threshold (1.0037). Consider maintaining minimal color scheme.")
        #     if not great_saturation:
        #         st.warning("⚠️ Saturation is above the recommended threshold (35.6761). Consider reducing vibrancy and choosing less vivid colors.")

        st.markdown("---")
else:
    if model_loaded:
        st.write("Upload an image to analyze its features and compare with optimal values for your target reach.")
    else:
        st.error("Model files not found. Please check that the model files exist in the 'models' directory.")
        st.error("Model files not found. Please check that the model files exist in the 'models' directory.")
