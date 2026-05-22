import streamlit as st
import requests
import os
from PIL import Image
import io
import plotly.graph_objects as go

# --- PAGE SETUP & HIGH-END STYLING ---
st.set_page_config(
    page_title="Teachable Machine Pro - Transfer Learning Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL configuration (handles Docker network and local running)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Premium CSS for custom dark theme, glassmorphism, and smooth animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Elegant Dark background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151821 50%, #0d0f14 100%);
        color: #e5e9f0;
    }
    
    /* Custom Card Style (Glassmorphism) */
    .glass-card {
        background: rgba(25, 30, 44, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Glowing Badges & Labels */
    .glow-text-cyan {
        color: #00e5ff;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        font-weight: bold;
    }
    .glow-text-purple {
        color: #bd93f9;
        text-shadow: 0 0 10px rgba(189, 147, 249, 0.5);
        font-weight: bold;
    }
    
    /* Custom Streamlit component modifications */
    .stButton>button {
        background: linear-gradient(135deg, #7b2cbf 0%, #9d4edd 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #9d4edd 0%, #c77dff 100%);
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(199, 125, 255, 0.6);
    }
    
    /* Custom metric styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        padding: 12px 18px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #00e5ff;
    }
    
    /* Sidebar aesthetic */
    [data-testid="stSidebar"] {
        background-color: #0c0e14;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown("""
<div style="text-align: center; padding: 15px 0 35px 0;">
    <h1 style="font-size: 3.2rem; background: linear-gradient(90deg, #00e5ff 0%, #7b2cbf 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        TEACHABLE MACHINE PRO
    </h1>
    <p style="font-size: 1.1rem; color: #a0aabf; font-weight: 300; letter-spacing: 0.5px;">
        Decoupled Client-Server Transfer Learning Engine • Powered by PyTorch & FastAPI
    </p>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "classes" not in st.session_state:
    st.session_state.classes = ["Class 1", "Class 2"]
if "trained" not in st.session_state:
    st.session_state.trained = False
if "system_status" not in st.session_state:
    st.session_state.system_status = {}

# --- HELPER FUNCTIONS TO INTERACT WITH BACKEND ---
def query_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            st.session_state.trained = status_data.get("is_model_trained", False)
            st.session_state.system_status = status_data
        else:
            st.sidebar.error("Error connecting to backend server API.")
    except Exception as e:
        st.sidebar.error(f"Cannot reach FastAPI server: {e}")

# Perform initial backend check
query_backend_status()

# --- SIDEBAR - SYSTEM DIAGNOSTICS & GLOBAL SETTINGS ---
with st.sidebar:
    st.markdown("### 📊 System Status Monitor")
    
    # Check if backend is alive
    try:
        ping = requests.get(f"{BACKEND_URL}/", timeout=3)
        if ping.status_code == 200:
            st.markdown('<div class="metric-container"><span>API Backend</span><span style="color: #00ff66; font-weight: bold;">● ONLINE</span></div>', unsafe_allow_html=True)
            engine_device = ping.json().get("engine_device", "unknown")
            st.markdown(f'<div class="metric-container"><span>ML Accelerator</span><span class="glow-text-cyan">{engine_device.upper()}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container"><span>API Backend</span><span style="color: #ff3333; font-weight: bold;">● OFFLINE</span></div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="metric-container"><span>API Backend</span><span style="color: #ff3333; font-weight: bold;">● UNREACHABLE</span></div>', unsafe_allow_html=True)

    # Dynamic status from state
    if st.session_state.system_status:
        st.markdown(f'<div class="metric-container"><span>Model Trained</span><span class="glow-text-purple">{"YES" if st.session_state.trained else "NO"}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-container"><span>Categories Loaded</span><span class="glow-text-cyan">{st.session_state.system_status.get("total_classes_defined", 0)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-container"><span>Total Samples</span><span class="glow-text-purple">{st.session_state.system_status.get("total_samples_collected", 0)}</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Reset System State")
    st.write("Wipe clean all saved classification samples on the server and delete trained weights.")
    if st.button("🚨 Reset Backend"):
        try:
            res = requests.post(f"{BACKEND_URL}/clear")
            if res.status_code == 200:
                st.success("All datasets and weights cleared successfully!")
                st.session_state.trained = False
                st.session_state.system_status = {}
                st.rerun()
            else:
                st.error("Failed to reset backend state.")
        except Exception as e:
            st.error(f"Error resetting: {e}")

# --- MAIN WORKFLOW GRID ---
col_classes, col_training, col_testing = st.columns([2.2, 1.2, 1.6])

# ==========================================
# COLUMN 1: DYNAMIC DATASET COLLECTION
# ==========================================
with col_classes:
    st.markdown("### 📁 1. Define Categories & Upload Samples")
    
    # Class manager controls
    class_mgmt_col1, class_mgmt_col2 = st.columns([2, 1])
    with class_mgmt_col1:
        new_class_input = st.text_input("New category name...", placeholder="e.g. Cup, Phone, Hand", label_visibility="collapsed")
    with class_mgmt_col2:
        if st.button("➕ Add Class", use_container_width=True) and new_class_input:
            clean_name = "".join(c for c in new_class_input if c.isalnum() or c in (" ", "-", "_")).strip()
            if clean_name and clean_name not in st.session_state.classes:
                st.session_state.classes.append(clean_name)
                st.rerun()
            elif clean_name in st.session_state.classes:
                st.warning("Category already exists.")
    
    st.write("Capture pictures from webcam or drag-and-drop folders below:")
    
    # Generate cards for each class
    for idx, class_name in enumerate(st.session_state.classes):
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin: 0; color: #00e5ff; display: inline;">Class {idx+1}: {class_name}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Action controls inside each class
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1:
            input_type = st.radio(f"Capture method for {class_name}:", ["Webcam Capture", "File Upload"], key=f"method_{class_name}")
        with c_col2:
            # Allow removing class if total classes > 2
            if len(st.session_state.classes) > 2:
                if st.button("🗑️ Remove Class", key=f"remove_{class_name}", use_container_width=True):
                    st.session_state.classes.remove(class_name)
                    st.rerun()

        # Input elements depending on radio choice
        uploaded_files = []
        if input_type == "File Upload":
            files = st.file_uploader(
                f"Drop images here for class '{class_name}'", 
                type=["png", "jpg", "jpeg", "webp"], 
                accept_multiple_files=True,
                key=f"uploader_{class_name}"
            )
            if files:
                uploaded_files = [f.getvalue() for f in files]
        else:
            # Webcam input
            captured_image = st.camera_input(f"Capture image for {class_name}", key=f"webcam_{class_name}")
            if captured_image:
                uploaded_files = [captured_image.getvalue()]
            else:
                with st.expander("📷 Camera not opening? Troubleshooting Guide"):
                    st.markdown("""
                    **1. Check Your Web Address (Secure Context)**
                    Browsers only allow camera access on **Secure Contexts**.
                    * **Correct:** Use **[http://localhost:8501](http://localhost:8501)** or **[http://127.0.0.1:8501](http://127.0.0.1:8501)**.
                    * **Incorrect:** Your network IP (e.g. `http://192.168.x.x:8501`) unless served over HTTPS.
                    
                    **2. Check Browser Site Permissions**
                    * Click the **Lock icon 🔒** in your browser address bar (left side of the URL).
                    * Make sure **Camera** is set to **Allow**.
                    
                    **3. Check for Device Conflict**
                    * Close other apps currently using your webcam (Zoom, Teams, or other browser tabs).
                    """)

        # Show current samples on the server for this class
        distribution = st.session_state.system_status.get("sample_distribution", {})
        server_sample_count = distribution.get(class_name, 0)
        
        st.markdown(f"**Samples on Server:** `{server_sample_count}`")
        
        # Upload trigger
        if len(uploaded_files) > 0:
            if st.button(f"📤 Upload {len(uploaded_files)} samples to '{class_name}'", key=f"upload_btn_{class_name}", use_container_width=True):
                with st.spinner("Uploading samples to server..."):
                    # Format as multipart form data
                    files_payload = []
                    for f_bytes in uploaded_files:
                        files_payload.append(('files', ('sample.jpg', f_bytes, 'image/jpeg')))
                    
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload-sample",
                            data={"class_name": class_name},
                            files=files_payload,
                            timeout=15
                        )
                        if res.status_code == 201:
                            st.success(f"Uploaded successfully! Class now has {res.json().get('total_class_samples')} samples.")
                            query_backend_status()
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {res.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# COLUMN 2: DYNAMIC TRAINING ENGINE
# ==========================================
with col_training:
    st.markdown("### 🧠 2. Deep Learning Train Engine")
    
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h4 style="margin: 0; color: #bd93f9;">Transfer Learning Configuration</h4>
        <p style="font-size: 0.9rem; color: #a0aabf; margin-top: 10px;">
            Extracting 576-dimensional semantic features using a pre-trained MobileNetV3 (Small) backbone and fitting a fast Logistic Regression on top.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic list of classes on the backend
    distribution = st.session_state.system_status.get("sample_distribution", {})
    
    st.write("**Dataset Diagnostics:**")
    if distribution:
        for c, count in distribution.items():
            st.text(f"• {c}: {count} samples")
    else:
        st.info("No training samples uploaded yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Extracting Deep Features & Training Classifier..."):
            try:
                res = requests.post(f"{BACKEND_URL}/train", timeout=60)
                if res.status_code == 200:
                    train_data = res.json()
                    if train_data.get("status") == "success":
                        st.success(train_data.get("message"))
                        st.balloons()
                        query_backend_status()
                        st.rerun()
                    else:
                        # Graceful error display from API validation
                        st.error(f"⚠️ Training Blocked:\n\n{train_data.get('message')}")
                else:
                    st.error(f"Backend returned HTTP {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to reach FastAPI backend: {e}")

    # Display Trained status card
    if st.session_state.trained:
        st.markdown("""
        <div style="background: rgba(0, 255, 102, 0.1); border: 1px solid #00ff66; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
            <span style="color: #00ff66; font-weight: bold; font-size: 1.1rem;">✓ Model Ready for Predictions</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(255, 153, 0, 0.1); border: 1px solid #ff9900; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
            <span style="color: #ff9900; font-weight: bold; font-size: 0.95rem;">⚠ Model not trained yet</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# COLUMN 3: LIVE PREVIEW & INFERENCE METER
# ==========================================
with col_testing:
    st.markdown("### 👁️ 3. Live Inference Dashboard")
    
    if not st.session_state.trained:
        st.info("The live inference dashboard is locked. Populate classes, upload samples, and press 'Train Model' above to unlock prediction features.")
    else:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin: 0; color: #00e5ff;">Test Interface</h4>
            <p style="font-size: 0.85rem; color: #a0aabf; margin: 5px 0 15px 0;">Feed instant image arrays and monitor live prediction probabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        test_method = st.radio("Inference Input Source:", ["Webcam Stream", "Image Uploader"], key="test_method")
        
        test_image_bytes = None
        if test_method == "Image Uploader":
            test_file = st.file_uploader("Upload an image for immediate classification:", type=["jpg", "jpeg", "png", "webp"])
            if test_file:
                test_image_bytes = test_file.getvalue()
                st.image(Image.open(io.BytesIO(test_image_bytes)), caption="Test Image Preview", use_container_width=True)
        else:
            captured_test = st.camera_input("Capture instant test image", key="test_webcam_input")
            if captured_test:
                test_image_bytes = captured_test.getvalue()
            else:
                with st.expander("📷 Camera not opening? Troubleshooting Guide"):
                    st.markdown("""
                    **1. Check Your Web Address (Secure Context)**
                    Browsers only allow camera access on **Secure Contexts**.
                    * **Correct:** Use **[http://localhost:8501](http://localhost:8501)** or **[http://127.0.0.1:8501](http://127.0.0.1:8501)**.
                    * **Incorrect:** Your network IP (e.g. `http://192.168.x.x:8501`) unless served over HTTPS.
                    
                    **2. Check Browser Site Permissions**
                    * Click the **Lock icon 🔒** in your browser address bar (left side of the URL).
                    * Make sure **Camera** is set to **Allow**.
                    
                    **3. Check for Device Conflict**
                    * Close other apps currently using your webcam (Zoom, Teams, or other browser tabs).
                    """)

        # Run Prediction If Image Available
        if test_image_bytes:
            with st.spinner("Processing image and predicting class probabilities..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/predict",
                        files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
                        timeout=10
                    )
                    
                    if res.status_code == 200:
                        pred_data = res.json()
                        if pred_data.get("status") == "success":
                            predicted_class = pred_data.get("predicted_class")
                            confidence = pred_data.get("confidence")
                            predictions = pred_data.get("predictions", {})
                            
                            st.markdown(f"**Predicted Category:** <span style='font-size: 1.4rem;' class='glow-text-cyan'>{predicted_class}</span> (Probability: `{confidence:.2%}`)", unsafe_allow_html=True)
                            
                            # Render highly polished horizontal bar chart of probabilities
                            y_labels = list(predictions.keys())
                            x_values = [predictions[k] * 100 for k in y_labels]
                            
                            # Elegant styling for Plotly Chart
                            fig = go.Figure(go.Bar(
                                x=x_values,
                                y=y_labels,
                                orientation='h',
                                marker=dict(
                                    color='#8a2be2',
                                    line=dict(color='#00e5ff', width=1.5)
                                ),
                                text=[f"{val:.1f}%" for val in x_values],
                                textposition='auto',
                                hoverinfo='x+y'
                            ))
                            
                            fig.update_layout(
                                title=None,
                                xaxis=dict(
                                    title="Confidence Score (%)", 
                                    range=[0, 100], 
                                    gridcolor='rgba(255, 255, 255, 0.05)', 
                                    zeroline=False,
                                    tickfont=dict(color='#a0aabf')
                                ),
                                yaxis=dict(
                                    gridcolor='rgba(255, 255, 255, 0.05)',
                                    tickfont=dict(color='#a0aabf', size=12)
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=40, r=20, t=10, b=40),
                                height=240,
                                font=dict(family="Outfit, sans-serif")
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.error(f"Inference Error: {pred_data.get('message')}")
                    else:
                        st.error(f"Prediction failed with server error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect for prediction: {e}")
