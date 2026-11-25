import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import bisindo_core as core

CONFIDENCE_THRESHOLD = 0.5
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = core.resource_path(str(BASE_DIR / "peraga_2_3_train_prominence_75.h5"))
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

app = FastAPI(title="BISINDO Sign Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def startup_event():
    global model
    model = core.load_classifier(MODEL_PATH)


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_FILE)


@app.get("/api/health")
def health_check():
    return {"ready": model is not None}


def extract_frames_from_video(video_path: str, mirror: bool) -> List:
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        frames.append(frame)

    cap.release()
    return frames


@app.post("/api/predict")
async def predict_video(
    video: UploadFile = File(...),
    mirror: bool = Form(True),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap dimuat.")

    suffix = Path(video.filename or "clip.webm").suffix or ".webm"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        frames = extract_frames_from_video(tmp_path, mirror=mirror)
        if not frames:
            raise HTTPException(status_code=400, detail="Video kosong atau gagal dibaca.")

        landmarks = core.frames_to_landmarks(frames)
        if landmarks is None:
            raise HTTPException(
                status_code=422,
                detail="Pose tubuh tidak terdeteksi. Pastikan bahu dan tangan terlihat.",
            )

        top3: List[Tuple[str, float]] = core.predict_top3(model, landmarks)
        formatted = [
            {"word": word, "prob": float(prob), "percent": round(float(prob) * 100, 1)}
            for word, prob in top3
        ]

        best = formatted[0]
        low_confidence = best["prob"] < CONFIDENCE_THRESHOLD

        return {
            "top3": formatted,
            "best_word": best["word"],
            "best_prob": best["prob"],
            "best_percent": best["percent"],
            "low_confidence": low_confidence,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat memproses: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass