---
title: WildFire AI
emoji: "🔥"
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: "1.46.1"
app_file: app.py
pinned: false
---

# WildFire AI

A multimodal deep learning system for wildfire detection and environmental risk prediction. The project combines computer vision, tabular machine learning, multimodal fusion, and explainable AI into a single interactive Streamlit application.

## Features

- Wildfire Detection using EfficientNet-B0
- Environmental Risk Prediction using XGBoost
- Multimodal Fusion of image and tabular data
- Grad-CAM visual explanations
- SHAP feature importance analysis
- Interactive Streamlit dashboard
- Global wildfire risk visualization with Folium

## Models

| Model | Framework |
|--------|-----------|
| CNN (EfficientNet-B0) | PyTorch |
| Risk Prediction | XGBoost |
| Fusion Model | PyTorch |

## Datasets

- DeepFire Dataset (1,900 images)
- UCI Forest Fires Dataset (517 records)

## Performance

| Model | Accuracy |
|--------|----------|
| CNN | 100% |
| XGBoost | 61.5% |
| Fusion | 100% |

## Technology Stack

- Python
- PyTorch
- XGBoost
- Streamlit
- Plotly
- Folium
- SHAP

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure

```
├── app.py
├── models/
├── preprocess.py
├── gradcam.py
├── spread_simulation.py
├── requirements.txt
├── packages.txt
└── README.md
```

## License

This project is intended for research and educational purposes.