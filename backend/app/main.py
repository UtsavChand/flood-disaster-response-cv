from fastapi import FastAPI, HTTPException, status

from backend.app.database import add_zone, initialize_database, list_zones
from backend.app.schemas import Zone
from pathlib import Path

import io
import logging

from fastapi import File, UploadFile
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOOD_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "unet_best.pth"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Nepal Flood Response API",
    version="1.0.0",
    description=(
        "Decision-support prototype. Recommendations require coordinator review; "
        "this API does not dispatch resources."
    ),
)


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "nepal-flood-response-api"}

@app.get("/api/v1/readiness")
def readiness():
    checkpoint_found = FLOOD_CHECKPOINT.is_file()

    return {
        "api": "available",
        "image_inference": (
            "checkpoint_found" if checkpoint_found else "checkpoint_missing"
        ),
        "checkpoint_found": checkpoint_found,
        "message": (
            "Checkpoint file exists; model loading has not been tested yet."
            if checkpoint_found
            else (
                "Image inference is unavailable. Obtain the trained "
                "checkpoints/unet_best.pth file before using image prediction."
            )
        ),
    }

@app.post("/api/v1/predictions/image")
async def predict_uploaded_image(file: UploadFile = File(...)):
    if not FLOOD_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image inference is unavailable: model checkpoint is missing.",
        )

    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as uploaded:
            if uploaded.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Upload a JPEG or PNG image.",
                )
            uploaded.verify()

        with Image.open(io.BytesIO(contents)) as uploaded:
            image = uploaded.convert("RGB")
            width, height = image.size

        from src.infer import predict_flood_mask

        mask = predict_flood_mask(image)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid JPEG or PNG image.",
        ) from exc
    except Exception as exc:
        logging.exception("Flood image inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flood image inference failed. Check the backend logs.",
        ) from exc

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "flood_fraction": float(mask.mean()),
        "georeferenced": False,
        "location": "unknown",
        "note": "Predicted flooded pixel share; not a mapped Nepal flood extent.",
    }


@app.get("/api/v1/zones", response_model=list[Zone])
def get_zones():
    return list_zones()


@app.post("/api/v1/zones", response_model=Zone, status_code=status.HTTP_201_CREATED)
def create_zone(zone: Zone):
    try:
        add_zone(zone)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Zone '{zone.id}' already exists",
            ) from exc
        raise

    return zone