# 🌿 Agronetics AI - Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Render](https://img.shields.io/badge/Render-Deployed-success.svg)

## 📌 Overview

**Agronetics AI** is an intelligent web application that uses deep learning to detect plant diseases from leaf images. The system can identify **66 different plant disease classes** across various crops including tomatoes, potatoes, grapes, apples, corn, and more with **98% accuracy**.

Farmers can simply take a photo of a plant leaf or upload an existing image, and the AI will instantly diagnose the disease, provide treatment recommendations, and suggest preventive measures.

## 🎯 Why Agronetics AI?

| Benefit | Description |
|---------|-------------|
| ⏱️ **Early Detection** | Identify diseases before they spread to entire crops |
| 💰 **Cost Effective** | Reduce crop losses by up to 60% and minimize treatment costs |
| 🌱 **Sustainable** | Get organic treatment recommendations for eco-friendly farming |
| 📊 **Data Driven** | Track disease patterns over time with scan history |
| 📱 **Easy to Use** | Simple camera interface - no technical expertise required |
| 🆓 **Free Access** | No subscription fees for basic disease detection |

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **AI Disease Detection** | 98% accuracy on test dataset with 66 disease classes |
| 📸 **Camera Upload** | Take photos directly from your phone camera |
| 🖼️ **Gallery Upload** | Upload existing images from your gallery or downloads |
| 🔐 **User Authentication** | Secure login and registration system |
| 📜 **Scan History** | Track all previous diagnoses with timestamps |
| 👤 **User Profile** | Personal information and scan statistics |
| 💊 **Treatment Guide** | Chemical and organic treatment options for each disease |
| 🌿 **Prevention Tips** | Best practices to avoid future disease outbreaks |
| 🧪 **Nutrient Analysis** | Identify and fix nutrient deficiencies in plants |
| 📱 **Mobile Responsive** | Works perfectly on phones, tablets, and desktops |
| 🌙 **Dark Mode** | Comfortable viewing in low-light conditions |

## 🎯 Supported Crops & Diseases

| Crop | Disease Classes |
|------|-----------------|
| 🍅 **Tomato** | Late Blight, Early Blight, Bacterial Spot, Septoria Leaf Spot, Leaf Mold, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Spider Mites, Healthy |
| 🥔 **Potato** | Late Blight, Early Blight, Healthy |
| 🍇 **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍎 **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌽 **Corn** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🫑 **Pepper** | Bacterial Spot, Healthy |
| 🍒 **Cherry** | Powdery Mildew, Healthy |
| 🍓 **Strawberry** | Leaf Scorch, Healthy |
| 🍑 **Peach** | Bacterial Spot, Healthy |
| 🫐 **Blueberry** | Healthy |
| 🍊 **Orange** | Huanglongbing (Citrus Greening) |
| 🥒 **Squash** | Powdery Mildew |

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework for routing and server management |
| **PyTorch** | Deep learning framework for disease detection |
| **ResNet50** | Pre-trained CNN architecture (transfer learning) |
| **TorchVision** | Image transformations and preprocessing |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure and semantics |
| **TailwindCSS** | Modern, responsive styling |
| **JavaScript** | Interactive features and API calls |
| **Material Symbols** | Beautiful icon system |

### Deployment
| Service | Purpose |
|---------|---------|
| **Render** | Cloud hosting platform |
| **Gunicorn** | Production WSGI server |
| **GitHub** | Version control and CI/CD |

## 📁 Project Structure

agronetics-ai/

├── app.py # Main Flask application

├── requirements.txt # Python dependencies

├── README.md # Project documentation

├── .gitignore # Git ignore file

├── templates/

├── index.html # Home page

├── login.html # Login page

├── scan.html # Camera/upload page

├── results.html # Diagnosis results

├── history.html # Scan history

└── profile.html # User profile

├── uploads/ # Temporary upload folder

└── best_cnn.pth # Trained model 


## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (optional)
- 4GB RAM minimum (8GB recommended)

⭐ Star History
If you find this project useful, please give it a star! It helps others discover the project.
