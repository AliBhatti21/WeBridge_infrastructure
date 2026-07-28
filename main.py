from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model import load_all_models, predict

# ── Global model store ─────────────────────────────────────────
# This dictionary holds the loaded models for the lifetime of the server.
# It is populated once at startup and reused on every request.
MODEL_STORE = {}

# ── Startup and shutdown ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield runs at startup
    print("\nLoading ensemble models...")
    MODEL_STORE['models'] = load_all_models()
    print("Server ready.\n")
    
    yield  # Server is now running and handling requests
    
    # Everything after yield runs at shutdown (cleanup)
    MODEL_STORE.clear()
    print("Models unloaded. Server shut down.")

# ── App instance ───────────────────────────────────────────────
app = FastAPI(
    title="WeBridge K2 Severity API",
    description="Ensemble inference for bridge damage severity classification (Low / Medium / High)",
    version="1.0.0",
    lifespan=lifespan
)

# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple liveness check.
    """
    return {
        "status": "ok",
        "models_loaded": len(MODEL_STORE.get('models', []))
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Upload a bridge inspection image.
    Returns K2 severity classification from the ensemble.
    
    - file: image file (JPEG, PNG, etc.)
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                    "Please upload a JPEG or PNG image."
        )

    # Read image bytes from the uploaded file
    image_bytes = await file.read()

    # Check models are loaded (safety guard)
    if 'models' not in MODEL_STORE:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Try again in a few seconds."
        )

    # Run inference
    try:
        result = predict(image_bytes, MODEL_STORE['models'])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    return JSONResponse(content=result)