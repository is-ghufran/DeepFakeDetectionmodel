# backend/app.py
import os
import shutil
import tempfile
import logging
import httpx # Using httpx for async requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from model_utils import load_and_prepare_model, load_video
from predict import predict_on_video

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(title="Deepfake Detection API")

# --- CORS (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model on Startup ---
try:
    logger.info("Loading model and feature extractor...")
    model, feature_extractor = load_and_prepare_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model, feature_extractor = None, None

# --- Gemini API Helper ---
async def get_gemini_explanation(prediction: str, confidence: float) -> str:
    """Calls the Gemini API to get a Gen-Z style explanation of the result."""
    # This is a simplified example. In a real app, manage your API key securely.
    api_key = "" # Canvas will provide this
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    percentage = round(confidence * 100)
    
    prompt = (
        f"You are a 'Gen-Z Explainer' AI. Your job is to explain the result of a deepfake detection test in a fun, modern, Gen-Z style. Use emojis and slang. "
        f"The result is: The video is {prediction}. "
        f"The confidence score is {percentage}%. "
        f"Explain what this means. If it's FAKE, be dramatic. If it's REAL, be reassuring. "
        f"For example, for FAKE you could say something like 'Yikes, this is giving major fake vibes! 💅 No cap, our AI is {percentage}% sure this ain't it.' "
        f"For REAL, you could say 'Okay, so we checked the vibes and it's looking legit. ✨ Our AI is {percentage}% sure this is the real deal. You're good.' "
        f"Now, generate your explanation for a {prediction} video with {percentage}% confidence."
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            result = response.json()
            
            if result.get('candidates'):
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                # Handle cases where the API response is unexpected
                logger.warning(f"Gemini API response missing candidates: {result}")
                return "Our AI is speechless, which is kinda weird. Try again maybe? 🤔"

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Gemini API: {e.response.text}")
        return "Oof, my brain is buffering. Couldn't get the hot take from the AI. 😵"
    except Exception as e:
        logger.error(f"An error occurred calling Gemini API: {e}")
        return "Something went sideways trying to get the tea. 😬"


# --- API Endpoints ---
@app.get("/", summary="Root endpoint to check API status")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Deepfake Detection API is running!"}


@app.post("/predict/", summary="Predict if a video is a deepfake")
async def predict_video(video_file: UploadFile = File(...)):
    """Accepts a video file, processes it, and returns the prediction."""
    if not model or not feature_extractor:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video_file.file, tmp)
            tmp_path = tmp.name
        
        logger.info(f"Processing video: {video_file.filename}")
        label, confidence = predict_on_video(tmp_path, model, feature_extractor)
        
        if label is None:
            raise HTTPException(status_code=400, detail="Could not process video.")
            
        logger.info(f"Prediction for {video_file.filename}: {label} (Confidence: {confidence:.4f})")
        
        return {
            "filename": video_file.filename,
            "prediction": label,
            "confidence_score": float(confidence)
        }

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        video_file.file.close()


@app.post("/analyze_verdict/", summary="Get a Gen-Z explanation for a prediction")
async def analyze_verdict(prediction: str, confidence: float):
    """Takes a prediction and confidence and gets an explanation from Gemini."""
    if not prediction or confidence is None:
        raise HTTPException(status_code=400, detail="Prediction and confidence score are required.")

    explanation = await get_gemini_explanation(prediction, confidence)
    
    return {"explanation": explanation}

