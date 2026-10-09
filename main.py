
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import re
from pathlib import Path


# A. Model paths
BASE_DIR = Path(__file__).resolve().parent
Model_path = BASE_DIR / "Artifacts" / "BiGRU_model.keras"

# B. Tokenizer path
Tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"

# C. Maximum sequence length
Max_Sequence_Length = 50

# D. Emotion labels
Emotions_Labels = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
]

# E. Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "🥹",
    "joy": "😃",
    "love": "😍",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮"
}


# 1. Text preprocessing
def preprocessing_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r",", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# 2. Request schema
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "hope you are doing well "
        }
    )


# 3. Prediction response schema
class PredictionResponse(BaseModel):
    text: str
    predicated_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


# 4. Health response schema
class HealthResponse(BaseModel):
    status: str

    model_loaded: bool


# 5. Load model and tokenizer at startup
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")

    if not Model_path.is_file():
        raise FileNotFoundError(f"Model not found: {Model_path}")

    if not Tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {Tokenizer_path}")

    dl_model["BiGRU"] = load_model(Model_path)

    with open(Tokenizer_path, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)

    print("Model loaded successfully!")

    yield

    # Clear model references when the server shuts down
    dl_model.clear()


# 6. Create FastAPI application
app = FastAPI(lifespan=lifespan)


# 7. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 8. Mount static files
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# 9. Homepage
@app.get("/", include_in_schema=False)
def server_ui():
    index_file = STATIC_DIR / "index.html"

    if not index_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="static/index.html not found"
        )

    return FileResponse(str(index_file))


# 10. Health check endpoint
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=(
            dl_model.get("BiGRU") is not None
            and dl_model.get("tokenizer") is not None
        )
    )


# 11. Prediction endpoint
# FIX: Use response_model= and TextInput as the request type
@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # Clean input text
    cleaned_text = preprocessing_text(text_input.text)

    if not cleaned_text:
        raise HTTPException(
            status_code=422,
            detail="Please enter valid text."
        )

    # FIX: Tokenizer method is texts_to_sequences()
    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequences = pad_sequences(
        tokenized_text,
        maxlen=Max_Sequence_Length,
        padding="post",
        truncating="post"
    )

    # Predict probabilities
    probabilities = BiGRU_model.predict(
        padded_sequences,
        verbose=0
    )[0]

    # FIX: argmax must use probabilities, not emotion labels
    top_emotion_index = int(np.argmax(probabilities))

    predicted_emotion = Emotions_Labels[top_emotion_index]

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(Emotions_Labels, probabilities)
    }

    # Return all fields required by PredictionResponse
    return PredictionResponse(
        text=text_input.text,
        predicated_emotion=predicted_emotion,
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )
