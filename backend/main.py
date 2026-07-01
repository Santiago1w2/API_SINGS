from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware



from backend.predictor import run_pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:300"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    if suffix not in [".mp4", ".webm", ".avi", ".mov"]:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        result = run_pipeline(path)
        return {
            "file": file.filename,
            "predictions": result
        }
    finally:
        os.unlink(path)