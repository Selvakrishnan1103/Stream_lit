import streamlit as st
import numpy as np
import joblib
import tempfile
from PIL import Image
from moviepy.editor import VideoFileClip
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

# Load your trained model
classification_model = joblib.load('classification_visual_only.pkl')

# Load ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title("🎓 Study Video Classifier (Visual Only)")
st.write("Upload a video to check if it’s **Educational** or **Non-Educational**.")

def extract_video_frames(video_path, interval=5):
    """Extract frames at every `interval` seconds."""
    video_clip = VideoFileClip(video_path)
    frames = []
    for t in range(0, int(video_clip.duration), interval):
        frame = video_clip.get_frame(t)
        frame_image = Image.fromarray(frame)
        frames.append(frame_image)
    video_clip.close()
    return frames

def extract_visual_features(video_path):
    """Extract average ResNet features from video frames."""
    frames = extract_video_frames(video_path)
    frame_features = []

    for frame in frames:
        frame = frame.resize((224, 224))
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array)
        frame_features.append(features.flatten())

    return np.mean(frame_features, axis=0)

# File uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    st.video(uploaded_video)
    st.info("⏳ Processing video... This may take a few seconds.")

    # Save the video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        temp_path = tmp_file.name

    try:
        visual_features = extract_visual_features(temp_path)
        prediction = classification_model.predict([visual_features])[0]

        st.success(f"✅ Prediction: **{'Educational' if prediction == 1 else 'Non-Educational'}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
