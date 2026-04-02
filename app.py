"""
MediCore AI — Streamlit Frontend
Production-grade medical AI interface
Author: Spandan Das
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import base64

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediCore AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    
    .main { background-color: #0a0f1e; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a0f1e 100%);
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #0099ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #8899aa;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .emergency-banner {
        background: linear-gradient(135deg, #ff0040, #ff4060);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        animation: pulse 1.5s infinite;
        margin: 1rem 0;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 64, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(255, 0, 64, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 64, 0); }
    }

    .emergency-text {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
    }

    .priority-1 {
        border-left: 4px solid #ff0040;
        background: rgba(255, 0, 64, 0.05);
    }
    .priority-2 {
        border-left: 4px solid #ff8800;
        background: rgba(255, 136, 0, 0.05);
    }
    .priority-3 {
        border-left: 4px solid #00ff88;
        background: rgba(0, 255, 136, 0.05);
    }

    .metric-card {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099ff) !important;
        color: #0a0f1e !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
    }

    .report-box {
        background: rgba(0, 212, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        color: #ccd9e8;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .scan-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .sidebar .sidebar-content {
        background: rgba(0,0,0,0.3) !important;
    }

    div[data-testid="stSidebarContent"] {
        background: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }

    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #e8f0fe !important;
    }

    .stRadio > div {
        gap: 0.5rem;
    }

    p, li { color: #aabbcc; }
    
    .divider {
        border: none;
        border-top: 1px solid rgba(0, 212, 255, 0.1);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        return r.json()
    except:
        return {}

def predict_symptoms(symptoms, generate_report=True):
    r = requests.post(
        f"{API_URL}/predict/symptoms",
        json={"symptoms": symptoms, "generate_report": generate_report},
        timeout=60
    )
    return r.json()

def predict_image(image_bytes, scan_type, generate_report=True):
    r = requests.post(
        f"{API_URL}/predict/image/{scan_type}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"generate_report": generate_report},
        timeout=60
    )
    return r.json()

def priority_color(p):
    return {1: "#ff0040", 2: "#ff8800", 3: "#00ff88"}.get(p, "#8899aa")

def priority_label(p):
    return {1: "🔴 CRITICAL", 2: "🟡 MODERATE", 3: "🟢 NON-URGENT"}.get(p, "UNKNOWN")

def get_gradcam(image_bytes, scan_type):
    try:
        r = requests.post(
            f"{API_URL}/gradcam/{scan_type}",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30
        )
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except:
        pass
    return None


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 800; 
                    background: linear-gradient(135deg, #00d4ff, #00ff88);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🏥 MediCore AI
        </div>
        <div style='color: #556677; font-size: 0.75rem; letter-spacing: 2px; 
                    text-transform: uppercase; margin-top: 0.3rem;'>
            Medical Intelligence
        </div>
    </div>
    <hr style='border-color: rgba(0,212,255,0.1);'>
    """, unsafe_allow_html=True)

    # API Status
    api_ok = check_api()
    status_color = "#00ff88" if api_ok else "#ff0040"
    status_text = "ONLINE" if api_ok else "OFFLINE"
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;'>
        <div style='width:8px; height:8px; border-radius:50%; background:{status_color}; 
                    box-shadow: 0 0 6px {status_color};'></div>
        <span style='color:{status_color}; font-size:0.75rem; font-weight:600; 
                     letter-spacing:1px;'>API {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    if api_ok:
        models = get_models()
        loaded = models.get("cnns", [])
        st.markdown("**Loaded Models:**")
        model_icons = {"chest": "🫁", "brain": "🧠", "skin": "🔬", "eye": "👁️"}
        for m in ["chest", "brain", "skin", "eye"]:
            icon = model_icons[m]
            ok = m in loaded
            color = "#00ff88" if ok else "#334455"
            st.markdown(f"<span style='color:{color};'>{'✅' if ok else '⬜'} {icon} {m.upper()}</span>", 
                       unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(0,212,255,0.1);'>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#556677; font-size:0.75rem;'>DQN: {'✅' if models.get('dqn') else '❌'} | Groq: {'✅' if models.get('groq') else '❌'} | {models.get('device','').upper()}</span>", 
                   unsafe_allow_html=True)
    else:
        st.error("Start the API: `py -3.12 main.py`")

    st.markdown("<hr style='border-color: rgba(0,212,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#556677; font-size:0.75rem; line-height:1.6;'>
    ⚠️ <b style='color:#8899aa;'>Disclaimer</b><br>
    MediCore AI is an AI-assisted tool. All diagnoses must be confirmed by a licensed physician.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">MediCore AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Multi-Modal Medical Intelligence System</div>', unsafe_allow_html=True)

if not api_ok:
    st.error("⚠️ API is offline. Please start the backend: `py -3.12 main.py`")
    st.stop()

# Mode selector
mode = st.radio(
    "Select Analysis Mode",
    ["🖼️ Medical Imaging", "🩺 Symptom Triage", "🔬 Combined Analysis"],
    horizontal=True
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 1: MEDICAL IMAGING
# ─────────────────────────────────────────────
if mode == "🖼️ Medical Imaging":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload Medical Scan")
        
        scan_type = st.selectbox(
            "Scan Type",
            options=["chest", "brain", "skin", "eye"],
            format_func=lambda x: {
                "chest": "🫁 Chest X-Ray (Pneumonia/Normal)",
                "brain": "🧠 Brain MRI (Tumor Detection)",
                "skin":  "🔬 Skin Lesion (Cancer Detection)",
                "eye":   "👁️ Retinal Scan (Diabetic Retinopathy)"
            }[x]
        )

        uploaded = st.file_uploader(
            "Upload image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        generate_report = st.checkbox("Generate AI Doctor Report (Groq LLM)", value=True)

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption=f"{scan_type.upper()} Scan", use_container_width=True)

        if uploaded and st.button("🔍 ANALYZE SCAN"):
            with st.spinner("Analyzing medical image..."):
                img_bytes = uploaded.getvalue()
                result = predict_image(img_bytes, scan_type, generate_report)

            with col2:
                st.markdown("### Analysis Results")

                if "error" in result:
                    st.error(result["error"])
                else:
                    cnn = result.get("cnn_result", {})
                    emergency = result.get("emergency", False)

                    # Emergency banner
                    if emergency:
                        st.markdown("""
                        <div class="emergency-banner">
                            <p class="emergency-text">🚨 EMERGENCY DETECTED — CALL 108 NOW!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Main prediction
                    conf = cnn.get("confidence", 0) * 100
                    pred = cnn.get("prediction", "Unknown")

                    st.markdown(f"""
                    <div class="result-card {'priority-1' if emergency else 'priority-3'}">
                        <div class="scan-badge">{scan_type.upper()} SCAN</div>
                        <div style='margin-top:1rem;'>
                            <div style='font-family: Syne, sans-serif; font-size: 1.8rem; 
                                        font-weight: 700; color: {"#ff0040" if emergency else "#00ff88"};'>
                                {pred}
                            </div>
                            <div style='color: #8899aa; font-size: 0.9rem; margin-top:0.3rem;'>
                                Confidence: <b style='color:#00d4ff;'>{conf:.1f}%</b>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # All scores
                    st.markdown("**Confidence Scores:**")
                    scores = cnn.get("all_scores", {})
                    for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
                        pct = score * 100
                        color = "#ff0040" if cls == pred and emergency else "#00d4ff" if cls == pred else "#334455"
                        st.markdown(f"""
                        <div style='display:flex; align-items:center; gap:0.8rem; margin:0.3rem 0;'>
                            <div style='width:120px; color:#8899aa; font-size:0.85rem;'>{cls}</div>
                            <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px;'>
                                <div style='width:{pct}%; background:{color}; height:8px; border-radius:4px;'></div>
                            </div>
                            <div style='width:50px; text-align:right; color:{color}; font-size:0.85rem; font-weight:600;'>{pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # GRAD-CAM
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("### 🔥 GRAD-CAM Heatmap")
                    with st.spinner("Generating attention heatmap..."):
                        gradcam_img = get_gradcam(img_bytes, scan_type)
                    if gradcam_img:
                        st.image(gradcam_img, caption="Red = High Attention | Blue = Low Attention", use_container_width=True)
                    else:
                        st.info("GRAD-CAM not available for this model")

                    # LLM Report
                    if generate_report and result.get("report"):
                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                        st.markdown("### 📋 AI Doctor Report")
                        st.markdown(f'<div class="report-box">{result["report"]}</div>', 
                                   unsafe_allow_html=True)

    if not uploaded:
        with col2:
            st.markdown("""
            <div style='height:300px; display:flex; align-items:center; justify-content:center;
                        border: 2px dashed rgba(0,212,255,0.15); border-radius:16px;
                        color: #334455; font-size:1rem; text-align:center;'>
                Upload a medical image<br>to see analysis results
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 2: SYMPTOM TRIAGE
# ─────────────────────────────────────────────
elif mode == "🩺 Symptom Triage":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Enter Patient Symptoms")

        # Common symptoms for easy selection
        common_symptoms = [
            "fever", "cough", "chest pain", "shortness of breath",
            "headache", "fatigue", "nausea", "vomiting", "dizziness",
            "abdominal pain", "back pain", "joint pain", "skin rash",
            "blurred vision", "loss of appetite", "weight loss",
            "sweating", "chills", "sore throat", "runny nose"
        ]

        selected = st.multiselect(
            "Select symptoms (or type custom):",
            options=common_symptoms,
            placeholder="Choose symptoms..."
        )

        custom = st.text_input(
            "Additional symptoms (comma-separated):",
            placeholder="e.g. chest tightness, dry cough"
        )

        if custom:
            extra = [s.strip() for s in custom.split(",") if s.strip()]
            selected = list(set(selected + extra))

        generate_report = st.checkbox("Generate AI Doctor Report", value=True)

        if selected:
            st.markdown(f"**Selected:** {', '.join(selected)}")

        if selected and st.button("🔍 ANALYZE SYMPTOMS"):
            with st.spinner("Running DQN triage analysis..."):
                result = predict_symptoms(selected, generate_report)

            with col2:
                st.markdown("### Triage Results")

                dqn = result.get("dqn_result", {})
                emergency = result.get("emergency", False)
                priority = dqn.get("priority", 2)

                # Emergency banner
                if emergency:
                    st.markdown("""
                    <div class="emergency-banner">
                        <p class="emergency-text">🚨 EMERGENCY — CALL 108 IMMEDIATELY!</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Priority card
                p_color = priority_color(priority)
                p_label = priority_label(priority)

                st.markdown(f"""
                <div class="result-card priority-{priority}">
                    <div style='font-family: Syne, sans-serif; font-size: 2.5rem; 
                                font-weight: 800; color: {p_color};'>
                        {p_label}
                    </div>
                    <div style='color:#8899aa; margin-top:0.5rem;'>
                        AI Triage Assessment via DQN Neural Network
                    </div>
                    <div style='margin-top:1rem; color:#ccd9e8;'>
                        <b>Matched symptoms:</b> {', '.join(dqn.get('matched_symptoms', [])) or 'None matched in training set'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Priority guide
                st.markdown("""
                <div style='margin-top:1rem;'>
                    <div style='color:#556677; font-size:0.8rem; margin-bottom:0.5rem; 
                                text-transform:uppercase; letter-spacing:1px;'>Priority Guide</div>
                    <div style='color:#ff0040; font-size:0.85rem;'>🔴 Priority 1 — Immediate emergency care</div>
                    <div style='color:#ff8800; font-size:0.85rem;'>🟡 Priority 2 — See doctor within 24 hours</div>
                    <div style='color:#00ff88; font-size:0.85rem;'>🟢 Priority 3 — Non-urgent, schedule appointment</div>
                </div>
                """, unsafe_allow_html=True)

                # LLM Report
                if generate_report and result.get("report"):
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("### 📋 AI Doctor Report")
                    st.markdown(f'<div class="report-box">{result["report"]}</div>',
                               unsafe_allow_html=True)

    if not selected:
        with col2:
            st.markdown("""
            <div style='height:300px; display:flex; align-items:center; justify-content:center;
                        border: 2px dashed rgba(0,212,255,0.15); border-radius:16px;
                        color: #334455; font-size:1rem; text-align:center;'>
                Select symptoms<br>to see triage results
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 3: COMBINED
# ─────────────────────────────────────────────
elif mode == "🔬 Combined Analysis":
    st.markdown("### Combined: Image + Symptoms → Full AI Report")

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.markdown("**1. Upload Scan**")
        scan_type = st.selectbox("Scan Type", ["chest", "brain", "skin", "eye"],
            format_func=lambda x: {"chest":"🫁 Chest","brain":"🧠 Brain","skin":"🔬 Skin","eye":"👁️ Eye"}[x])
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            st.image(Image.open(uploaded), use_container_width=True)

    with col2:
        st.markdown("**2. Add Symptoms**")
        common = ["fever","cough","chest pain","shortness of breath","headache",
                  "fatigue","nausea","dizziness","abdominal pain","blurred vision"]
        selected = st.multiselect("Symptoms", common, placeholder="Select...")
        custom = st.text_input("Custom symptoms", placeholder="comma-separated")
        if custom:
            selected = list(set(selected + [s.strip() for s in custom.split(",") if s.strip()]))

    with col3:
        st.markdown("**3. Run Analysis**")
        generate_report = st.checkbox("AI Doctor Report", value=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if uploaded and selected and st.button("🔬 RUN FULL ANALYSIS"):
            with st.spinner("Running combined analysis..."):
                img_bytes = uploaded.getvalue()
                # Run both separately and combine
                img_result = predict_image(img_bytes, scan_type, False)
                sym_result = predict_symptoms(selected, False)

                emergency = img_result.get("emergency") or sym_result.get("emergency")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Results
            if emergency:
                st.markdown("""
                <div class="emergency-banner">
                    <p class="emergency-text">🚨 EMERGENCY — CALL 108!</p>
                </div>
                """, unsafe_allow_html=True)

            rcol1, rcol2 = st.columns(2)
            with rcol1:
                cnn = img_result.get("cnn_result", {})
                pred = cnn.get("prediction","Unknown")
                conf = cnn.get("confidence",0)*100
                st.markdown(f"""
                <div class="result-card">
                    <div class="scan-badge">CNN RESULT</div>
                    <div style='font-family:Syne,sans-serif; font-size:1.4rem; 
                                font-weight:700; color:#00d4ff; margin-top:0.5rem;'>{pred}</div>
                    <div style='color:#8899aa;'>Confidence: {conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with rcol2:
                dqn = sym_result.get("dqn_result", {})
                priority = dqn.get("priority", 2)
                p_color = priority_color(priority)
                st.markdown(f"""
                <div class="result-card">
                    <div class="scan-badge">DQN TRIAGE</div>
                    <div style='font-family:Syne,sans-serif; font-size:1.4rem; 
                                font-weight:700; color:{p_color}; margin-top:0.5rem;'>
                        {priority_label(priority)}</div>
                    <div style='color:#8899aa;'>Based on symptoms</div>
                </div>
                """, unsafe_allow_html=True)

            # Generate combined report
            if generate_report:
                with st.spinner("Generating doctor report..."):
                    combined = requests.post(
                        f"{API_URL}/predict/symptoms",
                        json={"symptoms": selected, "generate_report": True},
                        timeout=60
                    ).json()
                if combined.get("report"):
                    st.markdown("### 📋 AI Doctor Report")
                    st.markdown(f'<div class="report-box">{combined["report"]}</div>',
                               unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#334455; font-size:0.8rem; padding:1rem 0;'>
    MediCore AI — Built by <b style='color:#00d4ff;'>Spandan Das</b> | 
    ResNet50 CNNs + DQN Triage + Groq LLM | 
    <b style='color:#ff4060;'>Not a substitute for professional medical advice</b>
</div>
""", unsafe_allow_html=True)