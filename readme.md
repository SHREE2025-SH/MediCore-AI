# 🏥 MediCore AI — Multi-Modal Medical Intelligence System

> Production-grade medical AI combining ResNet50 CNNs, DQN Reinforcement Learning, GRAD-CAM explainability, and Groq LLM doctor reports.

![MediCore AI](https://img.shields.io/badge/MediCore-AI-00d4ff?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge)

---

## 🧠 Architecture

```
Patient Input
├── Option A: Symptoms → DQN Triage (RL)
├── Option B: Medical Image → ResNet50 CNN
└── Combined → Groq LLM Doctor Report → Priority 1/2/3 → Emergency? Call 108!
```

---

## 🔬 Models

| Model | Task | Accuracy |
|-------|------|----------|
| 🫁 Chest X-Ray CNN | Pneumonia / Normal | 93.11% test |
| 🧠 Brain MRI CNN | Glioma / Meningioma / No Tumor / Pituitary | 98.10% val |
| 🔬 Skin Cancer CNN | 9-class ISIC Classification | 72.54% val |
| 👁️ Eye CNN | Diabetic Retinopathy (5 stages) | 83.88% test |
| 🤖 DQN Triage | Symptom → Priority 1/2/3 | RL-trained |

All CNNs use **ResNet50 Transfer Learning** with:
- Phase 1: Frozen backbone, custom head training
- Phase 2: Full fine-tuning with differential learning rates
- Class-weighted loss for imbalanced datasets
- Early stopping with patience=5

---

## ✨ Features

- **Multi-modal diagnosis** — Upload any medical scan OR enter symptoms
- **GRAD-CAM heatmaps** — Visual explainability showing where AI focused
- **DQN triage** — Reinforcement learning prioritizes patient urgency
- **Groq LLM reports** — Professional doctor-style reports via llama-3.3-70b
- **Emergency detection** — Automatic "Call 108" banner for critical cases
- **Production API** — FastAPI backend with Swagger docs
- **Beautiful UI** — Dark-themed Streamlit interface

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/SHREE2025-SH/MediCore-AI.git
cd MediCore-AI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn streamlit groq python-dotenv opencv-python matplotlib pandas pillow
```

### 2. Setup Environment
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

### 3. Add Trained Models
Place your `.pth` files in `cnn/`:
- `cnn/chest_xray_resnet50.pth`
- `cnn/brain_resnet50.pth`
- `cnn/skin_resnet50.pth`
- `cnn/eye_resnet50.pth`

### 4. Run
**Terminal 1 — API:**
```bash
python main.py
```

**Terminal 2 — UI:**
```bash
streamlit run app.py
```

Open `http://localhost:8501` 🎉

---

## 🏗️ Project Structure

```
MediCore-AI/
├── app.py                    # Streamlit UI
├── main.py                   # FastAPI backend
├── medicore_inference.py     # Unified inference engine
├── gradcam.py                # GRAD-CAM visualization
├── symptoms2.csv             # DQN symptom list
├── cnn/
│   ├── chest_xray_trainer.py
│   ├── brain_trainer.py
│   ├── skin_trainer.py
│   ├── eye_trainer.py
│   ├── chest_xray_resnet50.pth  # (not in repo - train locally)
│   ├── brain_resnet50.pth
│   ├── skin_resnet50.pth
│   └── eye_resnet50.pth
└── .env                      # GROQ_API_KEY (not in repo)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/models` | Loaded models |
| POST | `/predict/symptoms` | DQN triage |
| POST | `/predict/image/{type}` | CNN diagnosis |
| POST | `/predict/combined/{type}` | Full pipeline |
| POST | `/gradcam/{type}` | GRAD-CAM heatmap |
| GET | `/symptoms/list` | Available symptoms |

Interactive docs at `http://localhost:8000/docs`

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, ResNet50 Transfer Learning
- **RL**: DQN (Deep Q-Network)
- **Explainability**: GRAD-CAM
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, torchvision

---

## 📊 Training Details

Datasets used:
- Chest X-Ray: [Kaggle Chest X-Ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (5,863 images)
- Brain MRI: [Kaggle Brain Tumor](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 images)
- Skin Cancer: [ISIC Skin Cancer](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) (2,357 images)
- Eye DR: [Diabetic Retinopathy](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) (3,662 images)

---

## ⚠️ Disclaimer

MediCore AI is an AI-assisted diagnostic tool for educational and research purposes. **All diagnoses must be confirmed by a licensed physician.** This system is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author

**Spandan Das** — ML Engineer, Pune  
- GitHub: [@SHREE2025-SH](https://github.com/SHREE2025-SH)
- LinkedIn: [linkedin.com/in/spandan-das-ml](https://linkedin.com/in/spandan-das-ml)

---

*Built with ❤️ and 9 hours of GPU training*