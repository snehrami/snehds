import streamlit as st
st.set_page_config(page_title="✨ AI Vision Studio", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO
from fpdf import FPDF
from database import save_analysis, get_history
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, GRU
import os
import pandas as pd

# Load MobileNetV2
@st.cache_resource
def load_default_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

# Load Custom Model if exists
@st.cache_resource
def load_custom_model():
    if os.path.exists("cnn_gru_model.h5"):
        try:
            return tf.keras.models.load_model("cnn_gru_model.h5")
        except:
            return None
    return None

default_model = load_default_model()
custom_model = load_custom_model()

# Category Mapping Function
def get_category(label):
    human_keywords = ["person", "man", "woman", "boy", "girl"]
    food_keywords = ["pizza", "burger", "cake", "apple", "banana", "food", "dish", "orange", "lemon", "strawberry", "bakery", "bread"]
    device_keywords = ["laptop", "mobile", "phone", "keyboard", "screen", "computer", "tv", "monitor", "mouse", "ipod", "speaker"]
    animal_keywords = ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "elephant", "bear", "lion", "tiger", "monkey", "animal", "retriever", "terrier", "spaniel", "collie", "poodle", "hound", "husky", "pug", "tabby", "hound", "owl", "hawk", "eagle", "deer", "fox", "wolf", "rabbit", "mouse", "rat", "frog", "snake", "turtle", "lizard"]

    label = label.lower()

    if any(word in label for word in human_keywords):
        return "Human"
    elif any(word in label for word in animal_keywords):
        return "Animal"
    elif any(word in label for word in food_keywords):
        return "Food"
    elif any(word in label for word in device_keywords):
        return "Device"
    else:
        return "Other"

# Unsplash Integration (Industry Grade feature)
def get_unsplash_image(query, access_key):
    url = f"https://api.unsplash.com/photos/random"
    headers = {"Authorization": f"Client-ID {access_key}"}
    params = {"query": query, "orientation": "landscape"}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        image_url = data['urls']['regular']
        
        img_response = requests.get(image_url)
        img_response.raise_for_status()
        img = Image.open(BytesIO(img_response.content))
        return img
    except Exception as e:
        st.error(f"Error fetching image from Unsplash: {e}")
        return None

def create_pdf(label, category, confidence):
    """Generates a PDF bytes object from the analysis text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Vision Assistant - Classifier Report", ln=True, align="C")
    pdf.ln(10)
    
    report = f"Detected Object: {label}\nCategory: {category}\nConfidence: {confidence}%"
    for line in report.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
            
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= UI =================
def load_css():
    st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%);
    }
    
    h1 {
        background: linear-gradient(45deg, #ffffff, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stTabs"] {
        background: rgba(255, 255, 255, 0.02);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
    }

    button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(14, 165, 233, 0.4) !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Uploader */
    section[data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(14, 165, 233, 0.4);
        border-radius: 12px;
        transition: all 0.3s;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(14, 165, 233, 0.05);
        border-color: #0ea5e9;
    }
</style>
    ''', unsafe_allow_html=True)

load_css()
st.title("✨ AI Vision Studio Pro")
st.markdown("<p style='color:#9ca3af; font-size:1.1rem;'>Professional Grade Image Classification & Custom Training Hub</p>", unsafe_allow_html=True)
st.markdown("**Created by Rami Sneh**")

# Tabs
tab_dash, tab1, tab2, tab3 = st.tabs(["📈 Central Dashboard", "🔍 Analyzer", "📜 History", "🎓 Train Mode"])

with tab_dash:
    st.header("📊 Ecosystem Overview")
    st.markdown("Real-time metrics and intelligence gathering from your AI Vision system.")
    
    # Aggregate stats
    records = get_history(limit=5000)
    total = len(records)
    cat_counts = {"Human": 0, "Animal": 0, "Food": 0, "Device": 0, "Other": 0}
    conf_sum = 0
    valid = 0
    
    for r in records:
        try:
            parts = r.full_report.split("|")
            cat = parts[1].split(":")[1].strip()
            conf = float(parts[2].split(":")[1].replace("%","").strip())
            if cat in cat_counts: 
                cat_counts[cat] += 1
            else:
                cat_counts["Other"] += 1
            conf_sum += conf
            valid += 1
        except:
            pass
            
    avg_conf = (conf_sum / valid) if valid > 0 else 0.0
    
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("🌍 Total Images Processed", total)
    dcol2.metric("🎯 Avg. Model Confidence", f"{avg_conf:.1f}%")
    dcol3.metric("🧠 System Architectures", 2)
    
    st.markdown("---")
    
    if total > 0:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown("### 📈 Category Distribution")
            df_cat = pd.DataFrame({
                "Category": list(cat_counts.keys()),
                "Count": list(cat_counts.values())
            }).set_index("Category")
            st.bar_chart(df_cat, color="#0ea5e9", height=300)
            
        with col_c2:
            st.markdown("### 🏆 Top Category")
            top_cat = max(cat_counts, key=cat_counts.get)
            st.metric(top_cat, f"{cat_counts[top_cat]} operations", delta="Leading Class")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info(f"**Insights:** Model primarily processes **{top_cat}** imagery. Average response confidence holds strong at **{avg_conf:.0f}%**.")
    else:
        st.info("Launch the Analyzer and start predicting to populate your beautiful dashboard!")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_option = st.radio("Select Model for Analysis", ["MobileNetV2 (Pre-Trained)", "Custom CNN+GRU (Trained)"])
    if model_option == "Custom CNN+GRU (Trained)" and custom_model is None:
        st.warning("⚠️ Custom model not found! Go to Train Mode to train it first.")
        model_option = "MobileNetV2 (Pre-Trained)"
        
    st.markdown("---")
    
    source_option = st.radio("Choose Image Source", ["Upload Image", "Search Unsplash"])
    if source_option == "Search Unsplash":
        api_key_unsplash = st.text_input("Unsplash Access Key", type="password", help="Get it from Unsplash Developers")

img = None

with tab1:
    if source_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            
    elif source_option == "Search Unsplash":
        query = st.text_input("Enter a topic (e.g. dog, pizza, laptop):")
        if st.button("Fetch from Unsplash", type="primary"):
            if not api_key_unsplash:
                st.warning("Please provide your Unsplash Access Key in the sidebar.")
            elif query:
                with st.spinner(f"Fetching a stunning '{query}' image..."):
                    img_fetched = get_unsplash_image(query, api_key_unsplash)
                    if img_fetched:
                        st.session_state['fetched_img'] = img_fetched
                        
        if 'fetched_img' in st.session_state:
            img = st.session_state['fetched_img']

    if img:
        st.markdown("---")
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.image(img, caption="Source Image to Analyze", use_container_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        raw_img_array = np.array(img_resized)

        if len(raw_img_array.shape) == 2:
            raw_img_array = np.stack((raw_img_array,)*3, axis=-1)
            
        if raw_img_array.shape[-1] == 4:
            raw_img_array = raw_img_array[:, :, :3]

        raw_img_array = np.expand_dims(raw_img_array, axis=0)

        # Prediction
        with st.spinner("Classifying..."):
            if model_option == "MobileNetV2 (Pre-Trained)":
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(raw_img_array.astype(np.float32))
                predictions = default_model.predict(img_array)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
                label = decoded[0][1]
                confidence = round(decoded[0][2] * 100, 2)
                category = get_category(label)
            else:
                img_array = raw_img_array.astype(np.float32) / 255.0
                predictions = custom_model.predict(img_array)[0]
                classes = ['Animal', 'Device', 'Food', 'Human']
                max_idx = np.argmax(predictions)
                category = classes[max_idx]
                label = f"{category} (Custom)"
                confidence = round(predictions[max_idx] * 100, 2)

        # Output
        with col2:
            st.markdown("### 📊 Inference Results")
            
            st.metric("Detected Object", label)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Category", category)
            with col_m2:
                st.metric("Confidence Level", f"{confidence}%")
            
            st.progress(int(confidence) / 100)

            # DB Logic
            if st.button("Save & Generate PDF"):
                report_str = f"Object: {label} | Category: {category} | Match: {confidence}%"
                save_analysis(source_option, report_str)
                
                pdf_bytes = create_pdf(label, category, confidence)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="vision_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

with tab2:
    st.header("📜 Recent Analyses")
    history_records = get_history(limit=10)
    
    if not history_records:
        st.info("No past analyses found. Try analyzing and saving an image!")
    else:
        for record in history_records:
            with st.expander(f"Analysis on {record.created_at.strftime('%Y-%m-%d %H:%M:%S')} (Source: {record.image_source})"):
                st.text(record.full_report)

with tab3:
    st.header("🎓 Train Custom Model (CNN + GRU)")
    st.markdown("Train a custom `CNN + GRU` architecture for perfect 4-category classification: **Human, Animal, Food, Device**.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        epochs = st.slider("Epochs", min_value=1, max_value=30, value=5)
    with col_t2:
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        
    use_dummy_data = st.checkbox("Simulate training (No actual image data needed, perfect for demonstration!)", value=True)
    
    if st.button("🚀 Start Training Routine", type="primary"):
        st.info("Initializing CNN+GRU Architecture...")
        
        # Build Model
        m = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Reshape((26 * 26, 128)),
            GRU(128, return_sequences=False),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dense(4, activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        st.code("Model Architecture:\nLayer 1: Conv2D (32, 3x3) -> MaxPooling2D\nLayer 2: Conv2D (64, 3x3) -> MaxPooling2D\nLayer 3: Conv2D (128, 3x3) -> MaxPooling2D\nLayer 4: Reshape -> GRU (128 units)\nLayer 5: Dense (256) -> Dropout (0.5)\nOutput: Dense (4, Softmax)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart()
        
        class StreamlitTrainCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.history_dict = {"loss": [], "accuracy": []}
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                self.history_dict["loss"].append(logs.get("loss", 0.0))
                self.history_dict["accuracy"].append(logs.get("accuracy", 0.0))
                
                df = pd.DataFrame(self.history_dict)
                chart.line_chart(df)
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {logs.get('loss'):.4f} | Acc: {logs.get('accuracy'):.4f}")
        
        if use_dummy_data:
            st.warning("Running on dynamically generated dummy data for demonstration mode.")
            x_train = np.random.rand(batch_size * 2, 224, 224, 3).astype('float32')
            y_train = tf.keras.utils.to_categorical(np.random.randint(0, 4, size=(batch_size * 2)), num_classes=4)
            m.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                      callbacks=[StreamlitTrainCallback()], verbose=0)
            tf.keras.backend.clear_session()
        else:
            st.error("Real data mode selected but no dataset uploaded. Please use 'Simulate training' for now.")
            st.stop()
            
        m.save("cnn_gru_model.h5")
        st.success("✅ Training perfect! Custom CNN+GRU model saved successfully as `cnn_gru_model.h5`.")
        st.info("🔄 Re-run the app or restart to select your trained custom model in the sidebar.")
        st.balloons()