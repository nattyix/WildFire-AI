# 🔥 WildFire-AI
### AI-Powered Wildfire Detection & Environmental Risk Prediction using Deep Learning, Machine Learning, and Explainable AI

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-orange?style=for-the-badge)

</p>

---

## 🌍 Overview

WildFire-AI is an intelligent wildfire monitoring system that combines **Computer Vision**, **Machine Learning**, **Multimodal Deep Learning**, and **Explainable AI (XAI)** to detect wildfire incidents and estimate environmental wildfire risk.

Unlike conventional wildfire detection systems that rely solely on images or environmental parameters, WildFire-AI integrates **satellite/fire images** with **weather and environmental attributes**, providing a more reliable and comprehensive wildfire assessment.

The application is deployed as an interactive **Streamlit dashboard**, enabling users to upload wildfire images, analyze environmental conditions, visualize predictions, and understand AI decisions through explainability techniques.

---

## 🚀 Live Demo

**Hugging Face Space**

👉 https://huggingface.co/spaces/WildFire-AI/WildFire-AI

---

# ✨ Key Features

🔥 Wildfire Image Classification using EfficientNet-B0

🌡️ Environmental Wildfire Risk Prediction using XGBoost

🧠 Multimodal AI combining image and tabular data

📈 Interactive Streamlit Dashboard

🗺️ Global Wildfire Risk Visualization

🎯 Confidence Score Prediction

📊 SHAP Explainability for tabular predictions

🔥 Grad-CAM Heatmaps for image interpretation

⚡ Fast real-time inference

☁️ Cloud deployment on Hugging Face Spaces

---

# 🧠 AI Pipeline

```
Wildfire Image
       │
       ▼
 EfficientNet-B0 CNN
       │
 Image Features
       │
       ▼
Fusion Network  ◄──────── Environmental Data
       │                    (Temperature, RH, Wind, Rain)
       ▼
 Wildfire Detection
       │
       ▼
 Explainability
 ├── Grad-CAM
 └── SHAP
```

---

# 🏗️ Project Architecture

```
                User Input
              ───────────────
               │          │
               │          │
        Fire Image     Weather Data
               │          │
               ▼          ▼
      EfficientNet-B0   XGBoost
               │          │
               └──────┬───┘
                      ▼
               Fusion Network
                      │
                      ▼
         Wildfire Risk Prediction
                      │
        ┌─────────────┴────────────┐
        ▼                          ▼
   Grad-CAM                  SHAP Analysis
```

---

# 📂 Project Structure

```
WildFire-AI/
│
├── app.py                     # Streamlit dashboard
├── preprocess.py              # Data preprocessing
├── gradcam.py                 # Grad-CAM visualization
├── spread_simulation.py       # Wildfire spread simulation
│
├── train_cnn.py               # CNN training
├── train_tabular.py           # XGBoost training
├── train_fusion.py            # Fusion model training
├── train_lstm.py              # LSTM training
│
├── models/
│   ├── cnn_fire.pth
│   ├── fusion_model.pth
│   ├── xgb_fire.pkl
│   ├── lstm_weather.pth
│   ├── spread_engine.pkl
│   ├── scaler.pkl
│   └── lstm_scaler.pkl
│
├── data/
│   └── tabular/
│       └── forestfires.csv
│
├── requirements.txt
├── packages.txt
└── README.md
```

---

# 📊 Machine Learning Models

| Model | Purpose |
|---------|---------|
| EfficientNet-B0 | Wildfire Image Classification |
| XGBoost | Environmental Fire Risk Prediction |
| Fusion Neural Network | Combines Image + Environmental Features |
| LSTM | Weather Pattern Learning |
| Spread Simulation Engine | Fire Spread Estimation |

---

# 📈 Explainable AI (XAI)

WildFire-AI emphasizes model transparency by incorporating Explainable AI techniques.

### 🔥 Grad-CAM

Visualizes the regions of an image responsible for wildfire detection.

✔ Highlights flames

✔ Smoke regions

✔ Burnt vegetation

✔ Heat signatures

---

### 📊 SHAP

Provides feature importance for environmental predictions.

Important parameters include:

- Temperature
- Relative Humidity
- Wind Speed
- Rainfall
- Fuel Moisture
- Weather Conditions

---

# 📁 Dataset

## 🖼️ Image Dataset

- DeepFire Dataset
- Approximately **1,900 wildfire images**
- Fire and Non-Fire classes

---

## 🌲 Environmental Dataset

Forest Fires Dataset

Contains environmental variables including:

- Temperature
- Relative Humidity
- Wind Speed
- Rain
- Fire Area

---

# ⚙️ Tech Stack

### Programming

- Python

### Deep Learning

- PyTorch
- Torchvision

### Machine Learning

- XGBoost
- Scikit-learn

### Data Processing

- NumPy
- Pandas

### Visualization

- Plotly
- Matplotlib
- Folium

### Explainability

- SHAP
- Grad-CAM

### Deployment

- Streamlit
- Hugging Face Spaces

---

# 📈 Model Performance

| Model | Performance |
|---------|------------|
| CNN (EfficientNet-B0) | 100% Accuracy |
| XGBoost | 61.5% Accuracy |
| Fusion Model | 100% Accuracy |

> **Note:** Performance metrics are based on the datasets used during project development.

---

# 🖥️ Installation

Clone the repository

```bash
git clone https://github.com/nattyix/WildFire-AI.git
```

Move into the project directory

```bash
cd WildFire-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🚀 Usage

1. Launch the Streamlit application.
2. Upload a wildfire image.
3. Enter environmental parameters.
4. Run prediction.
5. View wildfire probability.
6. Explore Grad-CAM heatmaps.
7. Analyze SHAP explanations.
8. Monitor wildfire risk through interactive visualizations.

---

# 🌍 Dashboard Highlights

✅ Wildfire Image Classification

✅ Environmental Risk Prediction

✅ AI Explainability

✅ Interactive Charts

✅ Fire Spread Simulation

✅ Confidence Scores

✅ Modern Dark-Themed UI

---

# 🔮 Future Improvements

- Satellite imagery integration
- Real-time weather API support
- Drone-based wildfire monitoring
- Multi-class wildfire severity prediction
- Time-series forecasting
- GIS-based wildfire spread simulation
- Mobile application
- Early warning notification system

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve WildFire-AI:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is intended for **research, educational, and academic purposes**.

---

# 👨‍💻 Authors

Developed by the **Natalia Mathews** **Limnisha Changkakati** 

If you found this project helpful, don't forget to ⭐ the repository!

---

## 🌟 Support

If you like this project,

⭐ Star the repository

🍴 Fork it

