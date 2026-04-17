from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.predict import HealthResponse, PredictRequest, PredictResponse
from app.services.classifier import TextMemeClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "data.csv"
ARTIFACT_DIR = BASE_DIR / "back-end" / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = TextMemeClassifier(data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR)
    classifier.initialize()
    app.state.classifier = classifier
    yield


app = FastAPI(
    title="Text Meme Detector API",
    version="0.1.0",
    description="用于毕业设计演示的文本模因识别后端服务",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_classifier() -> TextMemeClassifier:
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=500, detail="classifier not initialized")
    return classifier


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    classifier = get_classifier()
    return HealthResponse(
        status="ok",
        mode=classifier.mode,
        data_path=str(classifier.data_path),
        sample_count=classifier.sample_count,
        metrics=classifier.metrics,
    )


@app.get("/labels")
def get_labels() -> dict[str, dict[str, str]]:
    return {
        "labels": {
            "0": "正常文本",
            "1": "疑似有害文本",
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    classifier = get_classifier()
    result = classifier.predict(payload.text)
    return PredictResponse(
        label=result.label,
        label_name=result.label_name,
        score=result.score,
        message=result.message,
    )
