"""
MediCore AI — FastAPI Backend
Production-grade medical AI API
Author: Spandan Das
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io
import sys
from pathlib import Path
from PIL import Image
from gradcam import generate_gradcam_image, create_gradcam_figure
import io as io_module

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from medicore_inference import MediCoreEngine

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="MediCore AI",
    description="Production-grade multi-modal medical AI — CNN + DQN + Groq LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engine once at startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("[MediCore API] Loading inference engine...")
    engine = MediCoreEngine()
    print("[MediCore API] Ready!")


# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────
class SymptomsRequest(BaseModel):
    symptoms: list[str]
    patient_info: Optional[dict] = None
    generate_report: Optional[bool] = True

class CombinedRequest(BaseModel):
    symptoms: list[str]
    scan_type: str
    patient_info: Optional[dict] = None
    generate_report: Optional[bool] = True


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "MediCore AI is running!",
        "version": "1.0.0",
        "author": "Spandan Das",
        "endpoints": [
            "GET  /health",
            "GET  /models",
            "POST /predict/symptoms",
            "POST /predict/image/{scan_type}",
            "POST /predict/combined/{scan_type}",
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "engine_loaded": engine is not None}


@app.get("/models")
async def get_models():
    """Returns list of loaded models and their status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    return engine.available_models()


@app.post("/predict/symptoms")
async def predict_symptoms(request: SymptomsRequest):
    """
    Symptom-based triage using DQN.
    Returns priority level (1/2/3) and emergency flag.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")

    result = engine.predict_symptoms(
        symptoms=request.symptoms,
        generate_report=request.generate_report,
        patient_info=request.patient_info
    )
    return result


@app.post("/predict/image/{scan_type}")
async def predict_image(
    scan_type: str,
    file: UploadFile = File(...),
    generate_report: bool = True
):
    """
    Image-based diagnosis using CNN.
    scan_type: chest | brain | skin | eye
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    valid_types = ["chest", "brain", "skin", "eye"]
    if scan_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan_type. Must be one of {valid_types}"
        )

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = engine.predict_image(
        image=image,
        scan_type=scan_type,
        generate_report=generate_report
    )
    return result


@app.post("/predict/combined/{scan_type}")
async def predict_combined(
    scan_type: str,
    symptoms: str,          # comma-separated string
    file: UploadFile = File(...),
    generate_report: bool = True
):
    """
    Full pipeline: image + symptoms → CNN + DQN + Groq LLM report.
    symptoms: comma-separated string e.g. "fever,cough,chest pain"
    scan_type: chest | brain | skin | eye
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    # Parse symptoms
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    # Read image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = engine.predict_combined(
        image=image,
        scan_type=scan_type,
        symptoms=symptom_list,
        generate_report=generate_report
    )
    return result


@app.get("/symptoms/list")
async def get_symptoms_list():
    """Returns the full list of supported symptoms for DQN triage."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if engine.dqn.model is None:
        return {"symptoms": [], "count": 0}
    return {
        "symptoms": engine.dqn.symptoms,
        "count": len(engine.dqn.symptoms)
    }

@app.post("/gradcam/{scan_type}")
async def get_gradcam(
    scan_type: str,
    file: UploadFile = File(...)
):
    """Returns GRAD-CAM heatmap overlay as image."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if scan_type not in engine.cnns:
        raise HTTPException(status_code=400, detail=f"{scan_type} CNN not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    model = engine.cnns[scan_type].model
    cnn_result = engine.cnns[scan_type].predict(image)
    
    overlay, heatmap, _ = generate_gradcam_image(model, image)
    
    # Return overlay image as bytes
    buf = io_module.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png",
                           headers={"X-Prediction": cnn_result["prediction"],
                                   "X-Confidence": str(cnn_result["confidence"])})


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )