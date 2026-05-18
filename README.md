# 🌿 Agronetics AI - Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0-blue.svg)
![Render](https://img.shields.io/badge/Render-Deployed-success.svg)

# 📌 Overview

**Agronetics AI** is an intelligent web application that uses deep learning to detect plant diseases from leaf images. The system can identify **66 different plant disease classes** across various crops including tomatoes, potatoes, grapes, apples, corn, and more with **98% accuracy**.

#🎯 Why Agronetics AI?

- ⏱️ **Early Detection** - Identify diseases before they spread
- 💰 **Cost Effective** - Reduce crop losses and treatment costs
- 🌱 **Sustainable** - Get organic treatment recommendations
- 📊 **Data Driven** - Track disease patterns over time

#✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **AI Disease Detection** | 98% accuracy on test dataset |
| 📸 **Image Upload** | Upload or capture leaf photos |
| 🔐 **User Authentication** | Secure login and registration system |
| 📜 **Scan History** | Track all previous diagnoses |
| 👤 **User Profile** | Personal information and statistics |
| 💊 **Treatment Guide** | Chemical and organic treatment options |
| 🌿 **Prevention Tips** | Best practices to avoid diseases |
| 🧪 **Nutrient Analysis** | Identify and fix nutrient deficiencies |
| 📱 **Mobile Responsive** | Works on all devices |

#🎯 Supported Crops & Diseases

| Crop | Disease Classes |
|------|-----------------|
| 🍅 **Tomato** | Late Blight, Early Blight, Bacterial Spot, Septoria Leaf Spot, Leaf Mold, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |
| 🥔 **Potato** | Late Blight, Early Blight, Healthy |
| 🍇 **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍎 **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌽 **Corn** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🫑 **Pepper** | Bacterial Spot, Healthy |
| 🍒 **Cherry** | Powdery Mildew, Healthy |
| 🍓 **Strawberry** | Leaf Scorch, Healthy |
| 🍑 **Peach** | Bacterial Spot, Healthy |

#🛠️ Tech Stack

Backend
| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework |
| **PyTorch** | Deep learning model |
| **ResNet50** | Pre-trained CNN architecture |
| **TorchVision** | Image transformations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Structure |
| **TailwindCSS** | Styling |
| **JavaScript** | Interactivity |
| **Material Symbols** | Icons |

### Deployment
| Service | Purpose |
|---------|---------|
| **Render** | Hosting |
| **Gunicorn** | WSGI server |
| **GitHub** | Version control |

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

└── best_cnn.pth # Trained model (optional)

# 🚀 Installation

# Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (optional)

# Step 1: Clone the Repository

```bash
git clone https://github.com/Darshita2865/agronetics-ai.git
cd agronetics-ai
