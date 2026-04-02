"""
MediCore AI — Unified Inference Engine
Loads all 4 CNNs + DQN and exposes a single predict() interface
Author: Spandan Das
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(r"D:\MediCore_AI")
CNN_DIR  = BASE_DIR / "cnn"

MODEL_PATHS = {
    "chest": CNN_DIR / "chest_xray_resnet50.pth",
    "brain": CNN_DIR / "brain_resnet50.pth",
    "skin":  CNN_DIR / "skin_resnet50.pth",
    "eye":   CNN_DIR / "eye_resnet50.pth",
}
META_PATHS = {
    "chest": CNN_DIR / "chest_xray_meta.json",
    "brain": CNN_DIR / "brain_meta.json",
    "skin":  CNN_DIR / "skin_meta.json",
    "eye":   CNN_DIR / "eye_meta.json",
}
DQN_PATH      = BASE_DIR / "dqn_model.pth"
DISEASE_PATH  = BASE_DIR / "dia_3.csv"
SYMPTOMS_PATH = BASE_DIR / "symptoms2.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MediCore Inference] Device: {DEVICE}")

# ─────────────────────────────────────────────
#  IMAGE TRANSFORM
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─────────────────────────────────────────────
#  BUILD RESNET50 ARCHITECTURE
# ─────────────────────────────────────────────
def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


# ─────────────────────────────────────────────
#  CNN LOADER
# ─────────────────────────────────────────────
class CNNModel:
    def __init__(self, name: str):
        self.name = name
        meta_path  = META_PATHS[name]
        model_path = MODEL_PATHS[name]

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train {name} CNN first!")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.num_classes = self.meta["num_classes"]

        self.model = build_resnet50(self.num_classes)
        ckpt = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        print(f"  ✅ {name.upper()} CNN loaded | Classes: {self.class_names}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Returns top prediction + confidence scores for all classes."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = img_transform(image).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx   = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        top_conf  = float(probs[top_idx])

        all_scores = {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs)}

        return {
            "model":      self.name,
            "prediction": top_class,
            "confidence": round(top_conf, 4),
            "all_scores": all_scores,
        }


# ─────────────────────────────────────────────
#  DQN TRIAGE MODEL
# ─────────────────────────────────────────────
class DQNTriage:
    PRIORITY_MAP = {0: 1, 1: 2, 2: 3}  # 3 actions → priority 1/2/3

    def __init__(self):
        if not DQN_PATH.exists():
            print(f"  ⚠️  DQN model not found at {DQN_PATH}. Symptom triage disabled.")
            self.model = None
            return

        # Load symptom list
        import pandas as pd
        df = pd.read_csv(SYMPTOMS_PATH, header=None)
        all_symptoms = [s.strip().lower() for s in df.iloc[:, 0].tolist()]

        # DQN was trained with 35 symptoms — use only first 35
        self.symptoms = all_symptoms[:35]
        self.num_symptoms = 35

        # Rebuild DQN architecture (matches Meta OpenEnv training exactly)
        class DQNNet(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size)
                )
            def forward(self, x):
                return self.network(x)

        self.dqn_net = DQNNet(35, 3).to(DEVICE)

        ckpt = torch.load(DQN_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and "policy_net" in ckpt:
            self.dqn_net.load_state_dict(ckpt["policy_net"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.dqn_net.load_state_dict(ckpt["model_state_dict"])
        else:
            self.dqn_net.load_state_dict(ckpt)
        self.model = self.dqn_net
        self.model.eval()
        print(f"  ✅ DQN Triage loaded | {self.num_symptoms} symptoms")

    @torch.no_grad()
    def predict(self, symptom_list: list[str]) -> dict:
        """symptom_list: list of symptom strings the patient has."""
        if self.model is None:
            return {"error": "DQN model not loaded"}

        # Build binary symptom vector (35 dims)
        vec = np.zeros(35, dtype=np.float32)
        matched = []
        for s in symptom_list:
            s_lower = s.strip().lower()
            if s_lower in self.symptoms:
                idx = self.symptoms.index(s_lower)
                vec[idx] = 1.0
                matched.append(s_lower)

        tensor = torch.FloatTensor(vec).unsqueeze(0).to(DEVICE)
        q_vals = self.dqn_net(tensor).cpu().numpy()[0]
        action = int(np.argmax(q_vals))
        priority = self.PRIORITY_MAP.get(action, 2)

        return {
            "model":           "dqn_triage",
            "matched_symptoms": matched,
            "action":          action,
            "priority":        priority,
            "priority_label":  f"Priority {priority}",
            "emergency":       priority == 1,
        }


# ─────────────────────────────────────────────
#  GROQ LLM REPORT
# ─────────────────────────────────────────────
class GroqReporter:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("  ⚠️  GROQ_API_KEY not found in .env")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            print("  ✅ Groq LLM ready (llama-3.3-70b-versatile)")

    def generate_report(self, cnn_result: dict = None,
                        dqn_result: dict = None,
                        patient_info: dict = None) -> str:
        if self.client is None:
            return "Groq API not configured. Add GROQ_API_KEY to .env"

        # Build context string
        context_parts = []

        if cnn_result:
            context_parts.append(
                f"Medical imaging analysis ({cnn_result['model'].upper()} scan): "
                f"Detected '{cnn_result['prediction']}' with {cnn_result['confidence']*100:.1f}% confidence. "
                f"All findings: {cnn_result['all_scores']}"
            )

        if dqn_result and "priority" in dqn_result:
            context_parts.append(
                f"Symptom triage: Patient reported symptoms {dqn_result.get('matched_symptoms', [])}. "
                f"AI triage assigned {dqn_result['priority_label']}. "
                f"Emergency: {dqn_result['emergency']}"
            )

        if patient_info:
            context_parts.append(f"Patient info: {patient_info}")

        context = "\n".join(context_parts)

        prompt = f"""You are MediCore AI, an advanced medical assistant. Based on the following AI analysis results, provide a clear, professional medical report for the attending physician.

{context}

Please provide:
1. Summary of findings
2. Possible diagnosis or differential diagnosis
3. Recommended next steps
4. Urgency level and whether emergency care is needed

Keep the report concise, professional, and actionable. Always remind that this is AI-assisted analysis and final diagnosis must be confirmed by a licensed physician.
Do NOT use markdown formatting, asterisks, bold, or bullet points. Write in plain text only."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────
#  EMERGENCY DETECTION
# ─────────────────────────────────────────────
EMERGENCY_CONDITIONS = {
    "chest": ["PNEUMONIA"],
    "brain": ["glioma", "meningioma", "pituitary"],
    "skin":  ["Melanoma", "Basal Cell Carcinoma", "melanoma", "basal cell carcinoma"],
    "eye":   ["Proliferate_DR", "Severe"],
}

def check_emergency(cnn_result: dict, dqn_result: dict = None) -> bool:
    model  = cnn_result.get("model", "")
    pred   = cnn_result.get("prediction", "")
    conf   = cnn_result.get("confidence", 0)

    cnn_emergency = (
        model in EMERGENCY_CONDITIONS and
        pred in EMERGENCY_CONDITIONS[model] and
        conf > 0.7
    )
    dqn_emergency = dqn_result.get("emergency", False) if dqn_result else False
    return cnn_emergency or dqn_emergency


# ─────────────────────────────────────────────
#  MEDICORE ENGINE  (main interface)
# ─────────────────────────────────────────────
class MediCoreEngine:
    def __init__(self, load_models: list[str] = None):
        """
        load_models: list of CNN names to load e.g. ["chest", "brain"]
                     Default: loads all available trained models
        """
        print("\n[MediCore] Initializing inference engine...")
        self.cnns     = {}
        self.dqn      = DQNTriage()
        self.reporter = GroqReporter()

        models_to_load = load_models or list(MODEL_PATHS.keys())
        for name in models_to_load:
            if MODEL_PATHS[name].exists():
                try:
                    self.cnns[name] = CNNModel(name)
                except Exception as e:
                    print(f"  ⚠️  Could not load {name}: {e}")
            else:
                print(f"  ⏳ {name.upper()} model not trained yet — skipping")

        print(f"\n[MediCore] Ready! Loaded CNNs: {list(self.cnns.keys())}\n")

    def predict_image(self, image: Image.Image, scan_type: str,
                      generate_report: bool = True,
                      patient_info: dict = None) -> dict:
        """
        scan_type: "chest" | "brain" | "skin" | "eye"
        Returns full prediction dict with optional LLM report.
        """
        if scan_type not in self.cnns:
            return {"error": f"{scan_type} CNN not loaded. Train it first!"}

        cnn_result = self.cnns[scan_type].predict(image)
        emergency  = check_emergency(cnn_result)

        result = {
            "scan_type":  scan_type,
            "cnn_result": cnn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                cnn_result=cnn_result,
                patient_info=patient_info
            )

        return result

    def predict_symptoms(self, symptoms: list[str],
                         generate_report: bool = True,
                         patient_info: dict = None) -> dict:
        """Symptom-based triage using DQN."""
        dqn_result = self.dqn.predict(symptoms)
        emergency  = dqn_result.get("emergency", False)

        result = {
            "symptoms":   symptoms,
            "dqn_result": dqn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                dqn_result=dqn_result,
                patient_info=patient_info
            )

        return result

    def predict_combined(self, image: Image.Image, scan_type: str,
                         symptoms: list[str],
                         generate_report: bool = True,
                         patient_info: dict = None) -> dict:
        """Full pipeline: image + symptoms → combined triage + LLM report."""
        cnn_result = self.cnns[scan_type].predict(image) if scan_type in self.cnns else None
        dqn_result = self.dqn.predict(symptoms) if symptoms else None
        emergency  = check_emergency(cnn_result, dqn_result) if cnn_result else \
                     (dqn_result.get("emergency", False) if dqn_result else False)

        result = {
            "scan_type":  scan_type,
            "cnn_result": cnn_result,
            "dqn_result": dqn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                cnn_result=cnn_result,
                dqn_result=dqn_result,
                patient_info=patient_info
            )

        return result

    def available_models(self) -> dict:
        return {
            "cnns":  list(self.cnns.keys()),
            "dqn":   self.dqn.model is not None,
            "groq":  self.reporter.client is not None,
            "device": str(DEVICE),
        }


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = MediCoreEngine()
    print("\n[Test] Available models:", engine.available_models())

    # Test symptom triage
    test_symptoms = ["fever", "cough", "chest pain", "shortness of breath"]
    print(f"\n[Test] Symptom triage for: {test_symptoms}")
    result = engine.predict_symptoms(test_symptoms, generate_report=False)
    print(f"  Priority: {result['dqn_result'].get('priority_label')}")
    print(f"  Emergency: {result['emergency']}")
    print(f"  Call 108: {result['call_108']}")