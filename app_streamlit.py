import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import io
import base64
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .nav-button {
        background-color: #17cf17;
        color: black;
        font-weight: bold;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: none;
        width: 100%;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .nav-button:hover {
        background-color: #15ba15;
    }
    .upload-area {
        border: 2px dashed #17cf17;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #f6f8f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 0.25rem;
        overflow: hidden;
        height: 1rem;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background-color: #17cf17;
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)
@app.route("/")
def home():
    return render_template("index.html")
# Load deficiency data
@st.cache_data
def load_deficiency_data():
    try:
        df = pd.read_csv('deficiency.csv')
        return df
    except Exception as e:
        st.error(f"Failed to load deficiency data: {e}")
        return None

deficiency_data = load_deficiency_data()

def get_recommendation(disease_class, confidence):
    """Get recommendation and estimate severity based on disease and confidence"""
    if deficiency_data is not None:
        # Find matching disease in deficiency data
        match = deficiency_data[deficiency_data['Disease'] == disease_class]
        if not match.empty:
            row = match.iloc[0]
            deficiency = row['Deficiency']
            advice = row['Advice']
            warning = row['Warning']
            
            # Estimate severity based on confidence
            if confidence >= 0.90:
                severity = "🔴 Severe"
                urgency = "Immediate action required"
            elif confidence >= 0.75:
                severity = "🟡 Moderate"
                urgency = "Treat within 2-3 days"
            elif confidence >= 0.60:
                severity = "🟠 Mild"
                urgency = "Monitor and treat within a week"
            else:
                severity = "🟢 Low"
                urgency = "Monitor closely"
            
            return {
                'deficiency': deficiency,
                'advice': advice,
                'warning': warning,
                'severity': severity,
                'urgency': urgency,
                'confidence': confidence
            }
    
    # Default recommendation if no match found
    return {
        'deficiency': 'Unknown',
        'advice': 'Consult agricultural expert for specific treatment',
        'warning': 'Disease may affect crop yield',
        'severity': '🟡 Unknown',
        'urgency': 'Consult expert',
        'confidence': confidence
    }
model_path = 'best_cnn.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define plant disease classes - exact match from notebook
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cauliflower_Black_Rot', 'Cauliflower_Healthy', 'Cherry_(including__sour)_Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)__Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Cotton_Aphids', 'Cotton_Bacterial_Blight', 'Cotton_Healthy', 'Cotton_Powdery_Mildew', 'Cotton_Target_spot',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Jackfruit_Algal_Leaf_Spot', 'Jackfruit_Black_Spot', 'Jackfruit_Healthy',
    'Mango_Anthracnose', 'Mango_Bacterial_Canker', 'Mango_Cutting_Weevil', 'Mango_Die_Back', 'Mango_Gall_Midge', 'Mango_Healthy', 'Mango_Powdery_Mildew', 'Mango_Sooty_Mould',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Pumpkin_Bacterial_Leaf_Spot', 'Pumpkin_Downy_Mildew', 'Pumpkin_Healthy', 'Pumpkin_Mosaic_Disease', 'Pumpkin_Powdery_Mildew',
    'Rice_BrownSpot', 'Rice_Healthy', 'Rice_Hispa', 'Rice_LeafBlast',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Sugarcane_Bacterial_Blights', 'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two_spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Define the model architecture
class MyNN(nn.Module):
    def __init__(self, ip_features=3, num_classes=66):
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(ip_features, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model
@st.cache_resource
def load_model():
    try:
        model = MyNN(ip_features=3, num_classes=len(disease_classes))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Image preprocessing - match notebook setup
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_disease(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        predicted_class = disease_classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = []
        for i in range(3):
            top3_predictions.append({
                'class': disease_classes[top3_indices[i].item()],
                'confidence': top3_prob[i].item()
            })
        
        return predicted_class, confidence_score, top3_predictions
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

# Navigation
def navigation():
    st.sidebar.markdown("### 🌱 AgroScan")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
    if st.sidebar.button("📷 Scan", use_container_width=True):
        st.session_state.page = "scan"
    if st.sidebar.button("📜 History", use_container_width=True):
        st.session_state.page = "history"
    if st.sidebar.button("👤 Profile", use_container_width=True):
        st.session_state.page = "profile"
    if st.sidebar.button("📚 Disease Info", use_container_width=True):
        st.session_state.page = "disease-info"
    if st.sidebar.button("💊 Recommendations", use_container_width=True):
        st.session_state.page = "recommendations"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# Home page
def home_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.image("https://lh3.googleusercontent.com/aida-public/AB6AXuDPhH4YpscGlX8u7ZmXm7bRsDytSdW33G8wdy8zJBRhs1wmqC1PKV0rdgCuzP3bO7S-XK8rSqtk-JLKIa0z7YNVEmh1svWzwuLnBvzf7j5eNm4g4o09GZ9dLFEN0CJtENJWrdEEwG8q7qH-_lY-XbObIvBSh0JtRAjNugZj6KqJWRz8ovLMW_7lYBzc3_2PPmtlDFf5m7vABBrEnmEuxIGIviOZjJOYS0yZ0EdXLpVFizX31TJV_uFbcpyxeSPEJruRahkV8h2Gnd7r", width=200)
    st.markdown("# Welcome to AgroScan")
    st.markdown('<p style="font-size: 1.1rem; color: #666;">Enhancing precision agriculture for healthier crops. Detect plant diseases early and accurately with our advanced machine learning technology.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.page = "scan"
        st.rerun()

# Scan page
def scan_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📷 New Scan")
    st.markdown('<p style="color: #666;">Capture a new image or upload an existing one to identify potential plant diseases.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'gif'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔬 Analyze Image", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, top3 = predict_disease(image)
                
                if predicted_class:
                    # Get recommendation and severity
                    recommendation = get_recommendation(predicted_class, confidence)
                    
                    # Store result in history
                    scan_result = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'top3': top3,
                        'image': uploaded_file.name,
                        'severity': recommendation['severity'],
                        'deficiency': recommendation['deficiency']
                    }
                    st.session_state.scan_history.insert(0, scan_result)
                    
                    # Display results with enhanced information
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Detection Results")
                    
                    # Main prediction with severity
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Disease:** {predicted_class}")
                        st.markdown(f"**Deficiency:** {recommendation['deficiency']}")
                    with col2:
                        st.markdown(f"**{recommendation['severity']}**")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Confidence bar
                    st.markdown('<div class="confidence-bar"><div class="confidence-fill" style="width: {}%"></div></div>'.format(confidence * 100), unsafe_allow_html=True)
                    
                    # Urgency warning
                    st.markdown(f"⚠️ **Urgency:** {recommendation['urgency']}")
                    
                    # Actionable recommendations
                    st.markdown("### 💡 Recommended Actions")
                    st.markdown(f"📋 **Advice:** {recommendation['advice']}")
                    
                    # Warning section
                    st.markdown("### ⚠️ Warning")
                    st.markdown(f"🚨 **Risk:** {recommendation['warning']}")
                    
                    # Top 3 predictions
                    st.markdown("### 🏆 Top 3 Predictions")
                    for i, pred in enumerate(top3, 1):
                        pred_recommendation = get_recommendation(pred['class'], pred['confidence'])
                        st.markdown(f"{i}. **{pred['class']}** - {pred['confidence']:.2%} ({pred_recommendation['severity']})")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to analyze the image. Please try again.")

# History page
def history_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📜 Scan History")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.scan_history:
        st.info("No scan history yet. Start by scanning a plant image!")
    else:
        for i, scan in enumerate(st.session_state.scan_history):
            with st.expander(f"📅 {scan['timestamp']} - {scan['prediction']}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Disease:** {scan['prediction']}")
                    st.markdown(f"**Confidence:** {scan['confidence']:.2%}")
                    st.markdown(f"**Image:** {scan['image']}")
                    if 'deficiency' in scan:
                        st.markdown(f"**Deficiency:** {scan['deficiency']}")
                with col2:
                    if 'severity' in scan:
                        st.markdown(f"**{scan['severity']}**")
                
                st.markdown("**Top 3 Predictions:**")
                for j, pred in enumerate(scan['top3'], 1):
                    pred_recommendation = get_recommendation(pred['class'], pred['confidence'])
                    st.markdown(f"{j}. {pred['class']} - {pred['confidence']:.2%} ({pred_recommendation['severity']})")

# Profile page
def profile_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 👤 Profile")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌱 About AgroScan")
    st.markdown("""
    AgroScan is an advanced plant disease detection application that uses cutting-edge 
    machine learning technology to identify plant diseases quickly and accurately.
    
    **Features:**
    - 🎯 High accuracy disease detection
    - 📱 Mobile-friendly interface
    - 📜 Scan history tracking
    - 🔬 Detailed analysis results
    - 🌱 Support for multiple plant types
    """)
    
    st.markdown("### 📊 Statistics")
    if st.session_state.scan_history:
        total_scans = len(st.session_state.scan_history)
        st.metric("Total Scans", total_scans)
        
        # Most common prediction
        predictions = [scan['prediction'] for scan in st.session_state.scan_history]
        if predictions:
            most_common = max(set(predictions), key=predictions.count)
            st.metric("Most Common Detection", most_common)
    else:
        st.info("No scans performed yet")

# Disease info page
def disease_info_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📚 Disease Information")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌱 Supported Plant Diseases")
    st.markdown("""
    Our model can detect diseases in the following plants:
    
    **🍎 Apple:**
    - Apple Scab
    - Black Rot
    - Cedar Apple Rust
    - Healthy
    
    **🍒 Cherry:**
    - Powdery Mildew
    - Healthy
    
    **🌽 Corn:**
    - Cercospora Leaf Spot
    - Common Rust
    - Northern Leaf Blight
    - Healthy
    
    **🍇 Grape:**
    - Black Rot
    - Esca (Black Measles)
    - Leaf Blight
    - Healthy
    
    **🍑 Peach:**
    - Bacterial Spot
    - Healthy
    
    **🌶️ Pepper:**
    - Bacterial Spot
    - Healthy
    
    **🥔 Potato:**
    - Early Blight
    - Late Blight
    - Healthy
    
    **🍓 Strawberry:**
    - Leaf Scorch
    - Healthy
    
    **🍅 Tomato:**
    - Bacterial Spot
    - Early Blight
    - Late Blight
    - Leaf Mold
    - Septoria Leaf Spot
    - Spider Mites
    - Target Spot
    - Yellow Leaf Curl Virus
    - Mosaic Virus
    - Healthy
    """)

# Recommendations page
def recommendations_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 💊 Plant Disease Recommendations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if deficiency_data is not None:
        st.markdown("### 📋 Complete Disease and Deficiency Database")
        
        # Search functionality
        search_term = st.text_input("🔍 Search diseases...", placeholder="Enter disease name...")
        
        # Filter by deficiency type
        deficiency_filter = st.selectbox("🔬 Filter by deficiency:", 
                                     ["All"] + sorted(deficiency_data['Deficiency'].unique()))
        
        # Filter data
        filtered_data = deficiency_data.copy()
        if search_term:
            filtered_data = filtered_data[filtered_data['Disease'].str.contains(search_term, case=False, na=False)]
        if deficiency_filter != "All":
            filtered_data = filtered_data[filtered_data['Deficiency'] == deficiency_filter]
        
        if not filtered_data.empty:
            # Display results in a formatted way
            for _, row in filtered_data.iterrows():
                with st.expander(f"🌱 {row['Disease']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Deficiency:** {row['Deficiency']}")
                        st.markdown(f"**Status:** {'🟢 Healthy' if row['Deficiency'] == 'No deficiency' else '🟡 Attention Required'}")
                    with col2:
                        st.markdown(f"**💡 Advice:** {row['Advice']}")
                        st.markdown(f"**⚠️ Warning:** {row['Warning']}")
        else:
            st.info("No diseases found matching your criteria.")
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Database Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Diseases", len(deficiency_data))
        with col2:
            healthy_count = len(deficiency_data[deficiency_data['Deficiency'] == 'No deficiency'])
            st.metric("Healthy Plants", healthy_count)
        with col3:
            disease_count = len(deficiency_data[deficiency_data['Deficiency'] != 'No deficiency'])
            st.metric("Diseases with Deficiencies", disease_count)
        
        # Deficiency distribution
        st.markdown("### 📈 Deficiency Distribution")
        deficiency_counts = deficiency_data['Deficiency'].value_counts()
        st.bar_chart(deficiency_counts)
        
    else:
        st.error("Deficiency data not available. Please ensure deficiency.csv is in the correct location.")

# Main app logic
navigation()

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "scan":
    scan_page()
elif st.session_state.page == "history":
    history_page()
elif st.session_state.page == "profile":
    profile_page()
elif st.session_state.page == "disease-info":
    disease_info_page()
elif st.session_state.page == "recommendations":
    recommendations_page()

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">🌱 AgroScan - Enhancing Precision Agriculture</p>', unsafe_allow_html=True)
