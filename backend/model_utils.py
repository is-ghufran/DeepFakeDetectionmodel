# backend/model_utils.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Configuration ---
# These should match the settings used during training
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20

def crop_center_square(frame):
    """Crops the center square of a frame."""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path: str, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    """Loads and preprocesses frames from a single video file."""
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    """Builds an EfficientNetB0 model for feature extraction."""
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = keras.applications.efficientnet.preprocess_input(inputs)
    outputs = base_model(preprocessed, training=False)
    return keras.Model(inputs, outputs, name="feature_extractor")

def load_and_prepare_model(model_path="deepfake_detection_model_final(new).h5"):
    """Loads the main Keras model and builds the feature extractor."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load the main sequence model
    # Note: If your model uses custom objects, you might need to provide a custom_objects dictionary
    main_model = keras.models.load_model(model_path)
    
    # Build the feature extractor (as it's not saved within the main model)
    feature_extractor = build_feature_extractor()
    
    return main_model, feature_extractor
