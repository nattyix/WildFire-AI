# app.py — WildfireAI Dashboard (Cloud Fixed)
import os, pickle, numpy as np, pandas as pd
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import matplotlib, matplotlib.cm as cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from preprocess import BASE
from streamlit.components.v1 import html

# Create required output directories on startup
os.makedirs(os.path.join(BASE, "outputs", "gradcam"), exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

st.set_page_config(
    page_title="WildfireAI", page_icon="🔥",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 20% 50%,
        #1a0a00 0%, #0a0a0f 40%, #000005 100%);
}
.block-container { padding: 1.5rem 2rem 2rem; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0800 0%, #0a0005 100%);
    border-right: 1px solid rgba(255,77,0,0.13);
}
[data-testid="stSidebar"] * { color: #ddd !important; }
.kpi-card {
    background: linear-gradient(135deg, #1a0800 0%, #0f0515 100%);
    border: 1px solid rgba(255,107,34,0.2);
    border-radius: 16px; padding: 1.4rem 1rem;
    text-align: center; position: relative;
    overflow: hidden; transition: transform .2s, border-color .2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(255,107,34,0.4); }
.kpi-card::before {
    content: ''; position: absolute;
    top: -40px; left: -40px; width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(255,77,0,0.08), transparent 70%);
    border-radius: 50%;
}
.kpi-val   { font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: .72rem; color: #888 !important;
             text-transform: uppercase; letter-spacing: .1em; margin-top: .3rem; }
.kpi-delta { font-size: .75rem; margin-top: .2rem; }
.alert-fire {
    background: linear-gradient(135deg, rgba(61,0,0,0.53), rgba(26,0,0,0.27));
    border: 1.5px solid rgba(255,77,0,0.8);
    border-radius: 12px; padding: 1.2rem;
    text-align: center; color: #ff6b22;
    font-weight: 700; font-size: 1.15rem;
    box-shadow: 0 0 24px rgba(255,77,0,0.2);
    animation: pulse-fire 2s infinite;
}
.alert-safe {
    background: linear-gradient(135deg, rgba(0,32,8,0.53), rgba(0,16,4,0.27));
    border: 1.5px solid rgba(34,197,94,0.53);
    border-radius: 12px; padding: 1.2rem;
    text-align: center; color: #4ade80;
    font-weight: 700; font-size: 1.15rem;
}
.alert-warn {
    background: linear-gradient(135deg, rgba(42,26,0,0.53), rgba(26,15,0,0.27));
    border: 1.5px solid rgba(245,158,11,0.53);
    border-radius: 12px; padding: 1.2rem;
    text-align: center; color: #fbbf24;
    font-weight: 700; font-size: 1.15rem;
}
@keyframes pulse-fire {
    0%,100% { box-shadow: 0 0 20px rgba(255,77,0,0.2); }
    50%      { box-shadow: 0 0 40px rgba(255,77,0,0.4); }
}
.sec-head {
    font-size: 1rem; font-weight: 600; color: #ff8c42;
    border-left: 3px solid #ff4d00; padding-left: .7rem;
    margin-bottom: 1.2rem; text-transform: uppercase; letter-spacing: .06em;
}
.risk-track {
    background: rgba(255,255,255,0.06); border-radius: 8px;
    height: 14px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.risk-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #ff4d00, #ff8c00);
    transition: width .6s cubic-bezier(.4,0,.2,1);
    box-shadow: 0 0 10px rgba(255,77,0,0.4);
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03); border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid rgba(255,77,0,0.13);
}
.stTabs [data-baseweb="tab"] {
    color: #777 !important; border-radius: 8px !important;
    padding: .4rem 1.1rem !important; font-size: .85rem !important;
    font-weight: 500 !important; transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff4d00, #cc2200) !important;
    color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff4d00, #cc2200) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: .6rem 2rem !important;
    font-weight: 600 !important; font-size: .9rem !important;
    box-shadow: 0 4px 15px rgba(255,77,0,0.27) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255,77,0,0.4) !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,77,0,0.13); border-radius: 12px;
    padding: .8rem 1rem;
}
[data-testid="stMetricValue"] { color: #ff8c42 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#ff4d00, #cc2200); border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────────────────────────
CNN_PATH    = os.path.join(BASE, "models", "cnn_fire.pth")
XGB_PATH    = os.path.join(BASE, "models", "xgb_fire.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
FUSION_PATH = os.path.join(BASE, "models", "fusion_model.pth")
SHAP_PATH   = os.path.join(BASE, "outputs", "shap_summary.png")
GRADCAM_DIR = os.path.join(BASE, "outputs", "gradcam")

# ── MODELS ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, 256), nn.ReLU(),
        nn.Dropout(p=0.2), nn.Linear(256, 2)
    )
    m.load_state_dict(torch.load(CNN_PATH, map_location='cpu'))
    m.eval(); return m

@st.cache_resource
def load_fusion():
    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(289,128), nn.BatchNorm1d(128),
                nn.ReLU(), nn.Dropout(.3),
                nn.Linear(128,64), nn.BatchNorm1d(64),
                nn.ReLU(), nn.Dropout(.2),
                nn.Linear(64,1), nn.Sigmoid()
            )
        def forward(self, x): return self.net(x).squeeze(1)
    m = FusionNet()
    m.load_state_dict(torch.load(FUSION_PATH, map_location='cpu'))
    m.eval(); return m

@st.cache_resource
def load_xgb():
    return (pickle.load(open(XGB_PATH,    'rb')),
            pickle.load(open(SCALER_PATH, 'rb')))

cnn_model          = load_cnn()
fusion_model       = load_fusion()
xgb_model, scaler = load_xgb()

# ── TRANSFORMS & HELPERS ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

def denormalize(t):
    m = torch.tensor([.485,.456,.406]).view(3,1,1)
    s = torch.tensor([.229,.224,.225]).view(3,1,1)
    return np.clip((t.squeeze(0)*s+m).permute(1,2,0).numpy(), 0, 1)

def get_embedding(t):
    f = cnn_model.features(t)
    f = cnn_model.avgpool(f)
    f = torch.flatten(f, 1)
    f = cnn_model.classifier[0](f)
    f = cnn_model.classifier[1](f)
    return cnn_model.classifier[2](f)

def run_gradcam(t):
    grads, acts = {}, {}
    h1 = cnn_model.features[-1].register_forward_hook(
        lambda m,i,o: acts.update({'f': o.detach()}))
    h2 = cnn_model.features[-1].register_full_backward_hook(
        lambda m,gi,go: grads.update({'f': go[0].detach()}))
    out  = cnn_model(t)
    pred = out.argmax(1).item()
    cnn_model.zero_grad()
    out[0, pred].backward()
    h1.remove(); h2.remove()
    w   = grads['f'].mean(dim=[2,3], keepdim=True)
    cam = torch.relu((w * acts['f']).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, (224,224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    probs = torch.softmax(out, 1)
    return cam, pred, probs[0][pred].item() * 100

def risk_hex(v):
    if v >= 75: return "#ff4d00"
    if v >= 50: return "#f59e0b"
    return "#22c55e"

def zcolor(r):
    if r >= 85: return '#ff2200'
    if r >= 75: return '#ff6600'
    if r >= 65: return '#f59e0b'
    return '#22c55e'

def build_tab_vec(temp, rh, wind, rain, ffmc, dmc, dc, isi):
    base   = ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
    months = [f'month_{x}' for x in
              ['apr','aug','dec','feb','jan','jul',
               'jun','mar','may','nov','oct','sep']]
    days   = [f'day_{x}' for x in
              ['fri','mon','sat','sun','thu','tue','wed']]
    eng    = ['temp_humidity_ratio','ffmc_isi_product',
              'dc_wind_interaction','dryness_score']
    cols   = base + months + days + eng
    v      = np.zeros((1, len(cols)))
    for i, val in enumerate([7,5,ffmc,dmc,dc,isi,temp,rh,wind,rain]):
        v[0,i] = val
    v[0, cols.index('temp_humidity_ratio')] = temp / (rh + 1)
    v[0, cols.index('ffmc_isi_product')]    = ffmc * isi
    v[0, cols.index('dc_wind_interaction')] = dc * wind
    v[0, cols.index('dryness_score')]       = ffmc + dmc + dc / 10
    return v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
        <div style='font-size:3rem'>🔥</div>
        <div style='font-size:1.4rem;font-weight:700;
                    background:linear-gradient(135deg,#ff8c42,#ff4d00);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent'>WildfireAI</div>
        <div style='font-size:.75rem;color:#666;margin-top:.2rem'>
            Multimodal Detection System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.8rem;color:#ff8c42;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.08em'>"
                "Models Active</div>", unsafe_allow_html=True)
    for name, score in [("EfficientNet-B0 CNN","100%"),
                         ("XGBoost Risk Model","61.5%"),
                         ("Fusion Network",    "100%")]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;
                    align-items:center;background:rgba(255,77,0,0.06);
                    border:1px solid rgba(255,77,0,0.13);border-radius:8px;
                    padding:.5rem .8rem;margin:.3rem 0'>
            <span style='font-size:.8rem;color:#ccc'>{name}</span>
            <span style='font-size:.8rem;font-weight:700;
                         color:#ff8c42'>{score}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.8rem;color:#ff8c42;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.08em'>"
                "Dataset</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.78rem;color:#888;line-height:1.9;margin-top:.4rem'>
        📸 DeepFire — 1,900 images<br>
        📊 UCI Forest Fires — 517 rows<br>
        🌍 25 zones across 6 continents<br>
        ⚖️ Balanced classes (50/50)
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#444;text-align:center;line-height:1.9'>
        PyTorch · XGBoost · SHAP<br>Grad-CAM · Folium · Plotly
    </div>""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 .5rem'>
    <div style='font-size:3rem;font-weight:800;line-height:1;
                background:linear-gradient(135deg,#ffffff 0%,
                #ff8c42 50%,#ff4d00 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        Wildfire Detection &amp; Risk Intelligence
    </div>
    <div style='color:#555;font-size:.95rem;margin-top:.6rem;font-weight:300'>
        EfficientNet-B0 · XGBoost · Multimodal Fusion ·
        Grad-CAM · SHAP · Global Coverage
    </div>
</div>""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    ("#ff4d00","100%",   "CNN Accuracy",    "+100% vs baseline"),
    ("#f59e0b","61.5%",  "Tabular Accuracy","UCI dataset ceiling"),
    ("#22c55e","100%",   "Fusion Accuracy", "Perfect fusion score"),
    ("#818cf8","1,900",  "Training Images", "DeepFire dataset"),
    ("#38bdf8","256-dim","CNN Embedding",   "Feature vector size"),
]
for col,(color,val,label,delta) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-val' style='color:{color}'>{val}</div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-delta' style='color:{color}'>{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs([
    "🖼️  Fire Detection",
    "📊  Risk Prediction",
    "🗺️  Global Risk Map",
    "📈  Model Insights"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — FIRE DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c_up, c_res = st.columns([1, 1.6], gap="large")
    with c_up:
        st.markdown("<div class='sec-head'>Upload Image</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=['jpg','jpeg','png'],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style='font-size:.78rem;color:#555;margin-top:.5rem;line-height:1.7'>
            Supports satellite imagery, aerial photos,<br>
            drone footage stills, or any forest scene.
        </div>""", unsafe_allow_html=True)

        if not uploaded:
            # Safe directory check — creates folder if missing, never crashes
            os.makedirs(GRADCAM_DIR, exist_ok=True)
            samples = [f for f in os.listdir(GRADCAM_DIR)
                       if f.endswith('_gradcam.png')][:2]
            if samples:
                st.markdown(
                    "<div style='margin-top:1rem;font-size:.78rem;"
                    "color:#666'>Sample outputs from test set:</div>",
                    unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                for col, f in zip([sc1,sc2], samples):
                    col.image(os.path.join(GRADCAM_DIR, f),
                              use_container_width=True)

    with c_res:
        if uploaded:
            img_bytes  = uploaded.read()
            img_pil    = Image.open(BytesIO(img_bytes)).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0)

            with st.spinner("🔍 Analyzing image..."):
                cam, pred, conf = run_gradcam(img_tensor)
                with torch.no_grad():
                    emb = get_embedding(img_tensor)
                tab_np        = np.zeros((1,33))
                tab_np[0,:10] = [7,5,85,30,200,5,25,40,10,0]
                tab_t         = torch.FloatTensor(tab_np)
                with torch.no_grad():
                    fusion_prob = fusion_model(
                        torch.cat([emb,tab_t],dim=1)).item()

            original_np = denormalize(img_tensor)
            heatmap     = cm.jet(cam)[:,:,:3]
            overlay     = np.clip(.55*original_np+.45*heatmap,0,1)

            p1,p2,p3 = st.columns(3, gap="small")
            p1.image(img_pil.resize((224,224)),
                     caption="Original", use_container_width=True)
            p2.image(Image.fromarray((overlay*255).astype(np.uint8)),
                     caption="Grad-CAM overlay", use_container_width=True)

            fig_cb, ax_cb = plt.subplots(figsize=(4,.35))
            fig_cb.patch.set_facecolor('none')
            ax_cb.set_facecolor('none')
            ax_cb.imshow(np.linspace(0,1,256).reshape(1,-1),
                         aspect='auto', cmap='jet')
            ax_cb.set_yticks([])
            ax_cb.set_xticks([0,128,255])
            ax_cb.set_xticklabels(['Low','Medium','High'],
                                   color='#888', fontsize=7)
            for sp in ax_cb.spines.values(): sp.set_visible(False)
            p2.pyplot(fig_cb, transparent=True)
            plt.close()

            with p3:
                label     = "🔥 FIRE DETECTED" if pred==1 else "✅ NO FIRE"
                css_class = "alert-fire"        if pred==1 else "alert-safe"
                st.markdown(f"<div class='{css_class}'>{label}</div>",
                            unsafe_allow_html=True)
                st.markdown("<div style='height:.8rem'></div>",
                            unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=conf,
                    number={"suffix":"%","font":{"color":"#ff8c42","size":28}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#444",
                                "tickfont":{"color":"#444","size":9}},
                        "bar":{"color":"#ff4d00","thickness":.25},
                        "bgcolor":"rgba(0,0,0,0)",
                        "steps":[
                            {"range":[0,40],"color":"rgba(34,197,94,0.15)"},
                            {"range":[40,70],"color":"rgba(245,158,11,0.15)"},
                            {"range":[70,100],"color":"rgba(255,77,0,0.15)"},
                        ],
                        "threshold":{"line":{"color":"#ff4d00","width":3},
                                     "thickness":.8,"value":conf}
                    },
                    title={"text":"CNN Confidence",
                           "font":{"color":"#888","size":11}}
                ))
                fig_g.update_layout(
                    height=180, margin=dict(l=20,r=20,t=30,b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={"family":"Inter"}
                )
                st.plotly_chart(fig_g, use_container_width=True)

                rcolor = risk_hex(fusion_prob*100)
                st.markdown(f"""
                <div style='margin:.5rem 0 .3rem;font-size:.78rem;color:#888;
                            text-transform:uppercase;letter-spacing:.06em'>
                    Fusion Risk Score</div>
                <div class='risk-track'>
                    <div class='risk-fill'
                         style='width:{fusion_prob*100:.0f}%;
                                background:linear-gradient(90deg,
                                {rcolor}aa,{rcolor})'></div>
                </div>
                <div style='text-align:right;font-size:.8rem;color:{rcolor};
                            margin-top:.3rem;font-weight:600'>
                    {fusion_prob*100:.1f}%</div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style='font-size:.73rem;color:#444;padding:.7rem;
                            background:rgba(255,255,255,0.02);
                            border:1px solid rgba(255,77,0,0.07);
                            border-radius:8px;margin-top:.8rem;line-height:1.7'>
                🔍 <b style='color:#ff8c42'>Grad-CAM</b> highlights regions
                the model attended to. Red = high activation, blue = low.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:320px;display:flex;align-items:center;
                        justify-content:center;
                        border:2px dashed rgba(255,77,0,0.13);
                        border-radius:16px;flex-direction:column;gap:1rem'>
                <div style='font-size:3rem'>🛰️</div>
                <div style='color:#444;font-size:.9rem'>
                    Upload an image to begin analysis</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='sec-head'>Weather & Vegetation Parameters</div>",
                unsafe_allow_html=True)
    col_s, col_r = st.columns([1.2,1], gap="large")

    with col_s:
        s1,s2 = st.columns(2)
        with s1:
            temp = st.slider("🌡️ Temperature °C", 0.0, 50.0, 28.0, .5)
            wind = st.slider("💨 Wind  km/h",      0.0, 30.0, 12.0, .5)
            ffmc = st.slider("🍂 FFMC",           18.7, 96.2, 88.0, .1)
            dc   = st.slider("🏜️ Drought Code",   7.9, 860.0,350.0, 1.)
        with s2:
            rh   = st.slider("💧 Humidity %",      0.0,100.0, 32.0, 1.)
            rain = st.slider("🌧️ Rain  mm",        0.0,  6.4,  0.0, .1)
            dmc  = st.slider("🌿 DMC",             1.1,291.3, 45.0, .5)
            isi  = st.slider("🌪️ ISI",             0.0, 56.1,  8.0, .1)
        predict_btn = st.button("🔥 Predict Fire Risk",
                                type="primary", use_container_width=True)

    with col_r:
        if predict_btn:
            vec      = build_tab_vec(temp,rh,wind,rain,ffmc,dmc,dc,isi)
            sc_vec   = scaler.transform(vec)
            xgb_prob = xgb_model.predict_proba(sc_vec)[0][1]
            risk_pct = xgb_prob * 100

            if risk_pct >= 65:
                st.markdown("<div class='alert-fire'>🔥 HIGH FIRE RISK</div>",
                            unsafe_allow_html=True)
            elif risk_pct >= 40:
                st.markdown("<div class='alert-warn'>⚠️ MODERATE RISK</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-safe'>✅ LOW RISK</div>",
                            unsafe_allow_html=True)
            st.markdown("<div style='height:.6rem'></div>",
                        unsafe_allow_html=True)

            rcolor = risk_hex(risk_pct)
            fig_r  = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(risk_pct,1),
                number={"suffix":"%","font":{"color":rcolor,"size":32}},
                delta={"reference":50,
                       "increasing":{"color":"#ff4d00"},
                       "decreasing":{"color":"#22c55e"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#333",
                            "tickfont":{"color":"#555","size":9}},
                    "bar":{"color":rcolor,"thickness":.3},
                    "bgcolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[0,40],"color":"rgba(34,197,94,0.12)"},
                        {"range":[40,65],"color":"rgba(245,158,11,0.12)"},
                        {"range":[65,100],"color":"rgba(255,77,0,0.12)"},
                    ],
                    "threshold":{"line":{"color":rcolor,"width":3},
                                 "thickness":.8,"value":risk_pct}
                },
                title={"text":"XGBoost Risk Score",
                       "font":{"color":"#888","size":12}}
            ))
            fig_r.update_layout(
                height=220, margin=dict(l=30,r=30,t=40,b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={"family":"Inter"}
            )
            st.plotly_chart(fig_r, use_container_width=True)

            factors = {
                "Temperature"  : temp/50,
                "Dryness"      : max(0,(50-rh)/50),
                "Wind"         : wind/30,
                "FFMC"         : (ffmc-18.7)/(96.2-18.7),
                "Drought (DC)" : min(dc/860,1),
                "Spread (ISI)" : min(isi/56,1),
            }
            fig_radar = go.Figure(go.Scatterpolar(
                r=list(factors.values()), theta=list(factors.keys()),
                fill='toself', fillcolor='rgba(255,77,0,0.15)',
                line=dict(color='#ff4d00',width=2),
                marker=dict(color='#ff8c42',size=6)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True,range=[0,1],
                                   color='#333',tickfont={"size":8}),
                    angularaxis=dict(color='#666',
                                     tickfont={"size":10,"color":"#aaa"})
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40,r=40,t=30,b=30),
                height=260, font={"family":"Inter","color":"#aaa"}
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.markdown("""
            <div style='height:420px;display:flex;align-items:center;
                        justify-content:center;
                        border:2px dashed rgba(255,77,0,0.13);
                        border-radius:16px;flex-direction:column;gap:1rem'>
                <div style='font-size:3rem'>🌡️</div>
                <div style='color:#444;font-size:.88rem;text-align:center'>
                    Adjust parameters and click<br>
                    <b style='color:#ff8c42'>Predict Fire Risk</b></div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — GLOBAL MAP
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sec-head'>Global Wildfire Risk Intelligence</div>",
                unsafe_allow_html=True)

    fire_zones = [
        {"name":"Paradise / Camp Fire",    "lat":39.75, "lon":-121.62,
         "risk":95,"acres":"153,336",    "year":2018,"region":"California, USA"},
        {"name":"Carr Fire — Shasta",      "lat":40.69, "lon":-122.39,
         "risk":91,"acres":"229,651",    "year":2018,"region":"California, USA"},
        {"name":"Mendocino Complex",       "lat":39.30, "lon":-123.35,
         "risk":88,"acres":"459,123",    "year":2018,"region":"California, USA"},
        {"name":"Caldor Fire — Tahoe",     "lat":38.79, "lon":-120.08,
         "risk":87,"acres":"221,835",    "year":2021,"region":"California, USA"},
        {"name":"Creek Fire — Fresno",     "lat":36.90, "lon":-119.50,
         "risk":83,"acres":"379,895",    "year":2020,"region":"California, USA"},
        {"name":"Glass Fire — Napa",       "lat":38.50, "lon":-122.26,
         "risk":82,"acres":"67,484",     "year":2020,"region":"California, USA"},
        {"name":"Black Summer — NSW",      "lat":-33.50,"lon":150.00,
         "risk":97,"acres":"12,000,000", "year":2020,"region":"New South Wales, AUS"},
        {"name":"Black Summer — Victoria", "lat":-37.20,"lon":148.00,
         "risk":93,"acres":"4,000,000",  "year":2020,"region":"Victoria, AUS"},
        {"name":"Canberra Fires",          "lat":-35.47,"lon":149.02,
         "risk":85,"acres":"629,000",    "year":2003,"region":"ACT, AUS"},
        {"name":"Wooroloo Fire — WA",      "lat":-31.80,"lon":116.30,
         "risk":78,"acres":"86,000",     "year":2021,"region":"W. Australia"},
        {"name":"Uttarakhand Fires",       "lat":30.07, "lon":79.45,
         "risk":84,"acres":"280,000",    "year":2021,"region":"Uttarakhand, India"},
        {"name":"Odisha Forest Fires",     "lat":20.50, "lon":84.80,
         "risk":76,"acres":"150,000",    "year":2023,"region":"Odisha, India"},
        {"name":"Mizoram — NE India",      "lat":23.16, "lon":92.94,
         "risk":72,"acres":"90,000",     "year":2021,"region":"Mizoram, India"},
        {"name":"Himachal Fires",          "lat":31.90, "lon":77.10,
         "risk":68,"acres":"60,000",     "year":2023,"region":"Himachal Pradesh"},
        {"name":"Greece — Attica",         "lat":38.20, "lon":23.80,
         "risk":89,"acres":"590,000",    "year":2021,"region":"Attica, Greece"},
        {"name":"Portugal — Pedrogao",     "lat":39.93, "lon":-8.15,
         "risk":86,"acres":"110,000",    "year":2017,"region":"Portugal"},
        {"name":"Turkey — Aegean Coast",   "lat":37.20, "lon":28.50,
         "risk":82,"acres":"400,000",    "year":2021,"region":"Turkey"},
        {"name":"Spain — Zamora",          "lat":41.80, "lon":-6.10,
         "risk":75,"acres":"74,000",     "year":2022,"region":"Castile, Spain"},
        {"name":"Amazon — Para",           "lat":-4.50, "lon":-54.00,
         "risk":92,"acres":"5,000,000",  "year":2019,"region":"Para, Brazil"},
        {"name":"Amazon — Mato Grosso",    "lat":-13.00,"lon":-56.00,
         "risk":90,"acres":"3,800,000",  "year":2020,"region":"Mato Grosso, Brazil"},
        {"name":"Pantanal — Bolivia",      "lat":-16.50,"lon":-62.00,
         "risk":85,"acres":"10,000,000", "year":2020,"region":"Bolivia"},
        {"name":"Fort McMurray — Alberta", "lat":56.72, "lon":-111.38,
         "risk":91,"acres":"1,500,000",  "year":2016,"region":"Alberta, Canada"},
        {"name":"BC Wildfires",            "lat":50.50, "lon":-121.00,
         "risk":87,"acres":"8,700,000",  "year":2023,"region":"British Columbia"},
        {"name":"Angola Savanna Fires",    "lat":-12.50,"lon":17.50,
         "risk":80,"acres":"20,000,000", "year":2019,"region":"Angola"},
        {"name":"Congo Basin",             "lat":-1.50, "lon":23.00,
         "risk":74,"acres":"8,000,000",  "year":2019,"region":"DRC"},
    ]

    m = folium.Map(location=[20.0,10.0], zoom_start=2,
                   tiles="CartoDB dark_matter")
    for z in fire_zones:
        c = zcolor(z['risk'])
        folium.CircleMarker(
            [z['lat'],z['lon']], radius=z['risk']/5.5,
            color=c, fill=True, fill_color=c,
            fill_opacity=.07, weight=0
        ).add_to(m)
        folium.CircleMarker(
            [z['lat'],z['lon']], radius=z['risk']/9,
            color=c, fill=True, fill_color=c,
            fill_opacity=.2, weight=.5
        ).add_to(m)
        folium.CircleMarker(
            [z['lat'],z['lon']], radius=7,
            color=c, fill=True, fill_color=c,
            fill_opacity=.95, weight=1.5,
            popup=folium.Popup(f"""
            <div style='font-family:Inter,sans-serif;background:#111;
                        padding:12px;border-radius:8px;min-width:190px'>
                <b style='color:{c};font-size:13px'>{z['name']}</b><br>
                <div style='color:{c};font-size:1.1rem;font-weight:700;
                            margin:5px 0'>Risk: {z['risk']}%</div>
                <div style='color:#aaa;font-size:11px;line-height:1.8'>
                    🌍 {z['region']}<br>📅 {z['year']}<br>
                    🔥 {z['acres']} acres</div>
            </div>""", max_width=230),
            tooltip=f"⚡ {z['name']} — {z['risk']}%"
        ).add_to(m)

    m.get_root().html.add_child(folium.Element("""
    <div style='position:fixed;bottom:28px;left:28px;z-index:9999;
                background:rgba(8,8,18,0.93);padding:14px 18px;
                border-radius:12px;border:1px solid rgba(255,77,0,0.2);
                font-family:Inter,sans-serif;color:white'>
        <div style='font-size:12px;font-weight:700;color:#ff8c42;
                    margin-bottom:8px;text-transform:uppercase;
                    letter-spacing:.06em'>Risk Level</div>
        <div style='font-size:11px;line-height:2.1'>
            <span style='color:#ff2200'>&#9899;</span> Extreme (85%+)<br>
            <span style='color:#ff6600'>&#9899;</span> Very High (75-84%)<br>
            <span style='color:#f59e0b'>&#9899;</span> High (65-74%)<br>
            <span style='color:#22c55e'>&#9899;</span> Moderate (below 65%)
        </div>
        <div style='font-size:10px;color:#444;margin-top:6px'>
            Click markers for details</div>
    </div>"""))

    map_html = m._repr_html_()
    html(map_html, height=520)


    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>Zone Summary — All Regions</div>",
                unsafe_allow_html=True)

    df_z = pd.DataFrame(fire_zones)[['name','region','risk','acres','year']]
    df_z.columns = ['Zone','Region','Risk %','Acres Burned','Year']
    df_z = df_z.sort_values('Risk %',ascending=False).reset_index(drop=True)

    fig_t = go.Figure(go.Table(
        columnwidth=[200,160,70,120,60],
        header=dict(
            values=['<b>Zone</b>','<b>Region</b>','<b>Risk %</b>',
                    '<b>Acres Burned</b>','<b>Year</b>'],
            fill_color='#1a0800',
            font=dict(color='#ff8c42',size=12,family='Inter'),
            line_color='#333', height=36, align='left'
        ),
        cells=dict(
            values=[df_z['Zone'],df_z['Region'],df_z['Risk %'],
                    df_z['Acres Burned'],df_z['Year']],
            fill_color=[
                ['#0f0800']*len(df_z),
                ['#0a0a0f']*len(df_z),
                ['#1a0800']*len(df_z),
                ['#0f0800']*len(df_z),
                ['#0a0a0f']*len(df_z),
            ],
            font=dict(
                color=['#ccc','#666',
                       [zcolor(r) for r in df_z['Risk %']],
                       '#aaa','#888'],
                size=12,family='Inter'
            ),
            line_color='#1a1a1a', height=30, align='left'
        )
    ))
    fig_t.update_layout(
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        height=len(df_z)*30+50
    )
    st.plotly_chart(fig_t, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sec-head'>Model Explainability & Architecture</div>",
                unsafe_allow_html=True)

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=["CNN Alone","XGBoost Alone","Fusion (Ours)"],
        y=[100, 61.54, 100],
        marker=dict(color=["#818cf8","#f59e0b","#ff4d00"],
                    line=dict(width=0)),
        text=["100%","61.54%","100%"],
        textposition='outside',
        textfont=dict(color='#ccc',size=13,family='Inter'),
        width=.45
    ))
    fig_perf.add_hline(
        y=95, line_dash="dash", line_color="#ff4d00",
        opacity=0.5, annotation_text="95% target",
        annotation_font_color="#ff8c42"
    )
    fig_perf.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        showlegend=False,
        margin=dict(l=40,r=20,t=20,b=20),
        yaxis=dict(
            range=[0,115],
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#555'),
            title=dict(text='Accuracy %',
                       font=dict(color='#666',size=11))
        ),
        xaxis=dict(
            tickfont=dict(color='#aaa',size=12),
            gridcolor='rgba(255,255,255,0.05)',
        )
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    i1,i2 = st.columns(2, gap="large")

    with i1:
        st.markdown("<div class='sec-head'>SHAP Feature Importance</div>",
                    unsafe_allow_html=True)
        if os.path.exists(SHAP_PATH):
            st.image(SHAP_PATH, use_container_width=True)
        else:
            st.info("Generating SHAP analysis live...")
            import shap
            import pandas as pd
            from preprocess import CSV_PATH
            df_s = pd.read_csv(CSV_PATH)
            df_s = pd.get_dummies(df_s, columns=['month','day'])
            df_s['label'] = (df_s['area'] > 0).astype(int)
            df_s['temp_humidity_ratio'] = df_s['temp']/(df_s['RH']+1)
            df_s['ffmc_isi_product']    = df_s['FFMC']*df_s['ISI']
            df_s['dc_wind_interaction'] = df_s['DC']*df_s['wind']
            df_s['dryness_score'] = df_s['FFMC']+df_s['DMC']+(df_s['DC']/10)
            X_s = df_s.drop(columns=['label'])
            X_sc = scaler.transform(X_s)
            explainer   = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_sc)
            fig_shap, ax_shap = plt.subplots(figsize=(10,6))
            fig_shap.patch.set_facecolor('#0f0800')
            shap.summary_plot(shap_values, X_s,
                      feature_names=list(X_s.columns),
                      show=False, max_display=12)
            st.pyplot(fig_shap)
            plt.close()

    with i2:
        st.markdown("<div class='sec-head'>Architecture Summary</div>",
                    unsafe_allow_html=True)
        arch = {
            "Component":["Image backbone","Embedding dim",
                          "Tabular model","Tabular features",
                          "Fusion input","Fusion hidden",
                          "Fusion output","Training images",
                          "Tabular rows"],
            "Detail":   ["EfficientNet-B0","256-dim vector",
                          "XGBoost (300 trees)","33 (incl. 4 engineered)",
                          "289-dim (256+33)","128 -> 64",
                          "Sigmoid (0-1 prob)","1,520 balanced",
                          "517 (UCI Portugal)"]
        }
        fig_arch = go.Figure(go.Table(
            header=dict(
                values=['<b>Component</b>','<b>Detail</b>'],
                fill_color='#1a0800',
                font=dict(color='#ff8c42',size=11,family='Inter'),
                line_color='#333', height=32
            ),
            cells=dict(
                values=[arch['Component'],arch['Detail']],
                fill_color=[['#0f0800','#0a0a0f']*5],
                font=dict(color=['#888','#ccc'],size=11,family='Inter'),
                line_color='#1a1a1a', height=28
            )
        ))
        fig_arch.update_layout(
            margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            height=310
        )
        st.plotly_chart(fig_arch, use_container_width=True)

        st.markdown("<div class='sec-head' style='margin-top:1rem'>"
                    "Grad-CAM Gallery</div>", unsafe_allow_html=True)

        # Safe directory check — never crashes if folder missing
        os.makedirs(GRADCAM_DIR, exist_ok=True)
        gc_files = [f for f in os.listdir(GRADCAM_DIR)
                    if f.endswith('_gradcam.png')][:3]
        if gc_files:
            gcols = st.columns(len(gc_files))
            for col,f in zip(gcols,gc_files):
                col.image(os.path.join(GRADCAM_DIR,f),
                          caption=f.replace('_gradcam.png','')[:14],
                          use_container_width=True)
        else:
            st.info("Grad-CAM gallery not available in cloud deployment. "
                    "Upload an image in the Fire Detection tab to generate live Grad-CAM.")