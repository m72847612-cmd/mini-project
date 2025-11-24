from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.inference import ForestInferenceService


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"

app = FastAPI(
    title="Deforestation Watch",
    description="Upload a forest satellite image to estimate deforestation levels.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

inference_service: ForestInferenceService | None = None


@app.on_event("startup")
def _startup() -> None:
    global inference_service
    inference_service = ForestInferenceService()


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def predict(
    threshold: float | None = Form(None),
    file: UploadFile = File(...),
) -> dict:
    if not inference_service:
        raise HTTPException(status_code=503, detail="Model is not ready yet, please retry shortly.")

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PNG or JPEG image.")

    file_bytes = await file.read()

    try:
        payload = inference_service.predict(
            file_bytes,
            filename=file.filename,
            threshold_override=threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected error while generating prediction.") from exc

    return {
        "filename": payload.filename,
        "deforestationRate": payload.deforestation_rate,
        "forestRate": payload.forest_rate,
        "deforestedPixels": payload.deforested_pixels,
        "forestPixels": payload.forest_pixels,
        "totalPixels": payload.total_pixels,
        "maskDataUrl": payload.mask_data_url,
        "overlayDataUrl": payload.overlay_data_url,
    }


