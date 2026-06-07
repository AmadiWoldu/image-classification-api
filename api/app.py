from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse

from api.schema import PredictionResponse, ImageRequest
from utils.logger import logger
from model.inference import predict_image
from model.preprocess import preprocess_image
from PIL import Image


logger.info("Image Classification API started")

app = FastAPI(
    title="Image Classification API",
    description="Predicts Image on CIFAR10",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & templates ──────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # points to project root

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ── Routes ────────────────────────────────────────────────

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Frontend",
    description="Serves the image classification web UI"
)
def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get(
    "/health",
    summary="Check Running Status",
    description="Checks whether the API is running or not"
)
def health():
    logger.info("Health Checkpoint Called")
    return {"status": "running"}


@app.post(
    "/predict",
    summary="Predict",
    description="Predicts the label of the image",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Prediction Request Received")

        # Validate file extension
        allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Open image once
        image = Image.open(file.file)

        result = predict_image(image)

        logger.info(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}")

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }

    except HTTPException:
        raise  # re-raise validation errors as-is

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))