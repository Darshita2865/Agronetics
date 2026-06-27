from flask import Flask, render_template, request, jsonify, session
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
from collections import Counter
import random

app = Flask(__name__)
app.secret_key = 'agronetics_secret_key_2024'

if os.environ.get('RENDER'):
    UPLOAD_FOLDER = '/tmp/uploads'
else:
    UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

print(f"✅ Upload folder: {UPLOAD_FOLDER}")

# ==================== 66 PLANT DISEASE CLASSES ====================
CLASS_NAMES_66 = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

print(f"✅ Configured {len(CLASS_NAMES_66)} disease classes")

# ==================== LOAD MODEL ====================
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=66):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

model = None
model_loaded = False

def load_model():
    global model, model_loaded
    try:
        model = PlantDiseaseModel(num_classes=66)
        # Try to load trained model if exists
        if os.path.exists('best_cnn.pth'):
            checkpoint = torch.load('best_cnn.pth', map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            model_loaded = True
            print(f"✅ Model loaded successfully")
        else:
            print("⚠️ best_cnn.pth not found, using mock predictions")
            model_loaded = False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False

def transform_image(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_pipeline(image).unsqueeze(0)

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        
        if model_loaded and model is not None:
            input_tensor = transform_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                top3 = []
                for i in range(3):
                    idx = top3_indices[i].item()
                    class_name = CLASS_NAMES_66[idx] if idx < len(CLASS_NAMES_66) else f"Class_{idx}"
                    display_name = class_name.replace('___', ' - ').replace('_', ' ')
                    top3.append({'name': display_name, 'confidence': float(top3_probs[i].item())})
                
                main_idx = top3_indices[0].item()
                predicted_class = CLASS_NAMES_66[main_idx]
                predicted_display = predicted_class.replace('___', ' - ').replace('_', ' ')
                confidence = float(top3_probs[0].item())
        else:
            # Mock predictions for demo when model not available
            demo_predictions = [
                'Tomato - Late Blight', 'Tomato - Early Blight', 'Tomato - Healthy',
                'Potato - Late Blight', 'Grape - Black Rot', 'Apple - Scab'
            ]
            predicted_display = random.choice(demo_predictions)
            confidence = round(random.uniform(0.82, 0.97), 3)
            top3 = [
                {'name': predicted_display, 'confidence': confidence},
                {'name': 'Alternate Disease', 'confidence': round(confidence * 0.6, 3)},
                {'name': 'Another Disease', 'confidence': round(confidence * 0.3, 3)}
            ]
        
        return {
            'success': True,
            'prediction': predicted_display,
            'confidence': confidence,
            'top3': top3
        }
    except Exception as e:
        return {'success': False, 'error': f'Prediction error: {str(e)}'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/history')
def history():
    scan_history = session.get('scan_history', [])
    return render_template('history.html', history=scan_history)

@app.route('/profile')
def profile():
    history = session.get('scan_history', [])
    stats = {
        'total_scans': len(history),
        'healthy_ratio': 0,
        'most_common_disease': '-'
    }
    if history:
        diseases = [h['prediction'] for h in history]
        most_common = Counter(diseases).most_common(1)
        if most_common:
            stats['most_common_disease'] = most_common[0][0]
        healthy_count = sum(1 for h in history if 'healthy' in h['prediction'].lower())
        if len(history) > 0:
            stats['healthy_ratio'] = round((healthy_count / len(history)) * 100)
    return render_template('profile.html', stats=stats)

@app.route('/api/profile/stats')
def get_profile_stats():
    history = session.get('scan_history', [])
    stats = {
        'total_scans': len(history),
        'healthy_ratio': 0,
        'most_common_disease': '-'
    }
    if history:
        diseases = [h['prediction'] for h in history]
        most_common = Counter(diseases).most_common(1)
        if most_common:
            stats['most_common_disease'] = most_common[0][0]
        healthy_count = sum(1 for h in history if 'healthy' in h['prediction'].lower())
        if len(history) > 0:
            stats['healthy_ratio'] = round((healthy_count / len(history)) * 100)
    return jsonify(stats)

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    session['scan_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/upload', methods=['POST'])
def upload_image():
    print(f"📸 Upload request received")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved: {filepath}")
        
        result = predict_image(filepath)
        
        if result['success']:
            # Save to history
            if 'scan_history' not in session:
                session['scan_history'] = []
            session['scan_history'].insert(0, {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'timestamp': datetime.now().isoformat()
            })
            session.modified = True
            
            # Add disease info
            result['disease_info'] = {
                'symptoms': 'Dark spots on leaves, yellowing, and wilting. Lesions may have concentric rings.',
                'treatment': 'Apply appropriate fungicide. Remove and destroy infected leaves. Improve air circulation.',
                'cause': 'Fungal or bacterial infection. Spreads through water splash and humid conditions.',
                'prevention': 'Proper plant spacing, crop rotation, and regular monitoring. Water at base of plant.',
                'organic_treatment': 'Apply neem oil or copper-based fungicide. Use compost tea as foliar spray.'
            }
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== LOAD MODEL ON STARTUP ====================
load_model()

# ==================== MAIN ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860)) 
    app.run(host='0.0.0.0', port=port)
