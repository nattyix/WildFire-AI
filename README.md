# WildFire-AI
A multimodal deep learning system for wildfire detection and risk prediction, combining computer vision, tabular ML, and explainable AI into an interactive dashboard.
## Features
- Fire Detection (EfficientNet-B0)
- Risk Prediction (XGBoost)
- Multimodal Fusion Model
- Grad-CAM Visualization
- SHAP Explainability
- Streamlit Dashboard
- Global Risk Map (Folium)

## Models
- CNN: EfficientNet-B0 (PyTorch)
- Tabular: XGBoost
- Fusion: Custom PyTorch Network

## Datasets
- DeepFire (1,900 images)
- UCI Forest Fires (517 rows)

## Performance
- CNN: 100%
- XGBoost: 61.5%
- Fusion: 100%

## Tech Stack
- PyTorch  
- XGBoost  
- Streamlit  
- Plotly  
- Folium  
- SHAP  

## Run
```bash
streamlit run app.py

