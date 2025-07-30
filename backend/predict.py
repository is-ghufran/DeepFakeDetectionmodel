# backend/predict.py
import numpy as np
from model_utils import load_video

# --- Configuration ---
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1280 # From EfficientNetB0

def predict_on_video(video_path: str, model, feature_extractor):
    """
    Runs a prediction on a single video file using the loaded models.
    """
    try:
        # 1. Load the video frames
        frames = load_video(video_path)
        if len(frames) == 0:
            print(f"Warning: Could not load frames from {video_path}")
            return None, None

        # 2. Extract features from the frames
        # The feature extractor expects a batch of frames
        features = feature_extractor.predict(frames, verbose=0)

        # 3. Prepare the data for the sequence model
        # The sequence model expects a batch of sequences, so we add a dimension
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")

        video_length = len(features)
        frame_features[0, :video_length] = features
        frame_mask[0, :video_length] = 1

        # 4. Make the final prediction
        prediction = model.predict([frame_features, frame_mask], verbose=0)[0]
        confidence = prediction[0]
        label = "FAKE" if confidence > 0.5 else "REAL"
        
        return label, confidence
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None
