import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
from PIL import Image

# --- Helper Functions ---

@st.cache_data  # Cache the paths to avoid re-scanning
def find_models():
    """Finds all 'best.pt' models in 'runs/detect/train*'."""
    base_path = Path.cwd()  # Assumes script is run from the project root
    detect_path = base_path / "runs" / "detect"
    model_paths = []
    
    if not detect_path.exists():
        return []
        
    # Use rglob to find all best.pt files within train* directories
    model_files = list(detect_path.rglob("train*/weights/best.pt"))
    
    for model_path in model_files:
        # Get the 'train*' folder name as the display name
        try:
            display_name = model_path.parent.parent.name
            model_paths.append((display_name, model_path))
        except:
            continue
            
    return model_paths

@st.cache_resource  # Cache the loaded model
def load_yolo_model(model_path):
    """Loads and caches a YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("📦 YOLO Object Detection App")

# --- Sidebar: Model Selection ---
st.sidebar.title("Model Selection")
model_files = find_models()

if not model_files:
    st.sidebar.error("No models found in 'runs/detect/'.")
    st.error("Could not find any trained models. Please make sure your 'runs/detect/train*/weights/best.pt' files are in the correct location.")
    st.stop()

# Create a dictionary for easy lookup: {display_name: path_object}
model_options = {name: path for name, path in model_files}

selected_model_name = st.sidebar.selectbox(
    "Choose a trained model:",
    options=model_options.keys(),
    help="Models are found in 'runs/detect/train*/weights/best.pt'"
)

# Load the selected model
selected_model_path = model_options[selected_model_name]
model = load_yolo_model(selected_model_path)

if model:
    st.sidebar.success(f"Loaded model: **{selected_model_name}**")
    st.sidebar.write(f"Path: `{selected_model_path}`")
else:
    st.stop()


# --- Main Page ---

# Tab 1: Live Prediction
tab1, tab2 = st.tabs(["📷 Live Prediction", "📊 Model Validation"])

with tab1:
    st.header("Predict on an Uploaded Image")
    st.info("Upload an image (jpg, jpeg, png) to see the model's predictions.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if uploaded_file is not None:
        # Read the image using PIL
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform prediction
        with st.spinner("Running prediction..."):
            results = model.predict(image, conf=conf_threshold)
            result = results[0]
            
            # Get annotated image
            img_annotated = result.plot()  # This is a numpy array (BGR)
            
            # Convert BGR (cv2) to RGB (PIL) for st.image
            img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(img_annotated_rgb, caption="Prediction Result", use_column_width=True)
            
        # Display detection details
        st.subheader("Detection Details")
        if len(result.boxes) == 0:
            st.warning("No objects detected with the current confidence threshold.")
        else:
            data_list = []
            for box in result.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                confidence = float(box.conf)
                x, y, w, h = box.xywhn[0].tolist() # Normalized
                
                data_list.append({
                    "Class": class_name,
                    "Confidence": f"{confidence:.2f}",
                    "x_center": f"{x:.4f}",
                    "y_center": f"{y:.4f}",
                    "width": f"{w:.4f}",
                    "height": f"{h:.4f}"
                })
            st.dataframe(data_list, use_container_width=True)


# Tab 2: Model Validation
with tab2:
    st.header(f"Validate Model: `{selected_model_name}`")
    
    # Find the yolo_params.yaml
    yaml_path = Path.cwd() / 'yolo_params.yaml'
    
    if not yaml_path.exists():
        st.error(f"Cannot find 'yolo_params.yaml' in the root directory.")
        st.warning("Please make sure 'yolo_params.yaml' is in the same folder as this app.")
    else:
        st.success(f"Found 'yolo_params.yaml'")
        
        with open(yaml_path, 'r') as file:
            try:
                data_params = yaml.safe_load(file)
                st.json(data_params, "Contents of 'yolo_params.yaml'")
            except Exception as e:
                st.error(f"Error reading YAML file: {e}")
                data_params = None

        if data_params and 'test' in data_params and data_params['test'] is not None:
            st.info(f"Test dataset path: **{data_params['test']}**")
            
            if st.button("Run Validation on Test Set", type="primary"):
                with st.spinner("Running validation... This may take some time."):
                    try:
                        # Run validation
                        metrics = model.val(data=str(yaml_path), split="test")
                        
                        st.subheader("Validation Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("mAP50-95 (Box)", f"{metrics.box.map:.4f}")
                        col2.metric("mAP50 (Box)", f"{metrics.box.map50:.4f}")
                        col3.metric("mAP75 (Box)", f"{metrics.box.map75:.4f}")
                        
                        st.subheader("All Metrics")
                        st.json(metrics.box.json) # Display all metrics
                        
                        st.subheader("Confusion Matrix")
                        cm_path = metrics.save_dir / "confusion_matrix.png"
                        if cm_path.exists():
                            st.image(str(cm_path), caption="Confusion Matrix")
                        else:
                            st.warning("Confusion matrix image not found.")

                        st.subheader("P-R Curve")
                        pr_path = metrics.save_dir / "PR_curve.png"
                        if pr_path.exists():
                            st.image(str(pr_path), caption="Precision-Recall Curve")
                        else:
                            st.warning("P-R curve image not found.")

                    except Exception as e:
                        st.error(f"An error occurred during validation:")
                        st.exception(e)
        else:
            st.warning("No 'test' field found in 'yolo_params.yaml'. Cannot run validation.")