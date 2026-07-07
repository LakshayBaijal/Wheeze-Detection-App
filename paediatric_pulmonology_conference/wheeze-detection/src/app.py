"""
Streamlit web app for testing the wheeze detection model.

Non-technical users (like Geetesh) can:
  1. Upload or record a breathing sound clip
  2. See the model's prediction instantly
  3. View confidence and medical metrics
  4. Save results

Run it:
  streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import librosa
import joblib
from pathlib import Path
from scipy.signal import butter, sosfilt
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# SETTINGS & PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"  # user test results saved here
try:
    RESULTS_DIR.mkdir(exist_ok=True)
except Exception:
    pass  # read-only cloud disk; result download still works

SAMPLE_RATE = 8000
CLIP_SECONDS = 2.0
BANDPASS_LOW_HZ = 80
BANDPASS_HIGH_HZ = 2000
CLIP_LEN = int(CLIP_SECONDS * SAMPLE_RATE)

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def _bandpass(signal, sr):
    """Keep only lung-sound frequencies (80-2000 Hz)."""
    nyquist = sr / 2.0
    low = BANDPASS_LOW_HZ / nyquist
    high = min(BANDPASS_HIGH_HZ / nyquist, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal)


def _fix_length(signal):
    """Cut or zero-pad to fixed length."""
    if len(signal) >= CLIP_LEN:
        return signal[:CLIP_LEN]
    padding = CLIP_LEN - len(signal)
    return np.pad(signal, (0, padding), mode="constant")


def clean_audio(audio_data, sr=SAMPLE_RATE):
    """Clean a full audio clip (not just an event slice)."""
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

    audio_data = _bandpass(audio_data, SAMPLE_RATE)

    # Normalise
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak

    return _fix_length(audio_data.astype(np.float32))


def extract_features(signal):
    """Extract MFCC features for the model."""
    N_MFCC = 40
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)
    return np.concatenate([means, stds]).astype(np.float32)


# ============================================================================
# MODEL LOADING & PREDICTION
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model, looking in a few likely places."""
    candidates = [
        MODEL_DIR / "model_random_forest.joblib",       # outputs/ folder
        PROJECT_ROOT / "model_random_forest.joblib",     # project top folder
        Path(__file__).resolve().parent / "model_random_forest.joblib",  # next to app.py
    ]
    for model_path in candidates:
        if model_path.exists():
            return joblib.load(model_path)
    return None


def predict_wheeze(audio_data, sr):
    """Predict wheeze vs no-wheeze on an audio clip."""
    model = load_model()
    if model is None:
        return None, None

    # Clean and extract features
    signal = clean_audio(audio_data, sr)
    features = extract_features(signal)
    features = features.reshape(1, -1)

    # Predict
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    return pred, prob


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="Wheeze Detection", layout="centered")

st.markdown("""
# 🫁 Wheeze Detection Test Tool

**For:** Dr. Geetesh Baijal — testing the wheeze detection model on patient recordings.

Simply upload or record a breathing sound clip, and the model will predict whether it contains wheeze.
""")

st.divider()

# --- Input section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Upload a file")
    uploaded_file = st.file_uploader(
        "Upload a .wav file (2 seconds of breathing)",
        type=["wav", "mp3", "ogg"]
    )

with col2:
    st.subheader("Option 2: Record now")
    audio_data = st.audio_input("🎙️ Record a clip (2 seconds)")

st.divider()

# --- Process audio ---
audio_to_process = None
sr = None

if uploaded_file is not None:
    try:
        audio_to_process, sr = librosa.load(uploaded_file, sr=None)
        st.success(f"✓ File loaded: {len(audio_to_process)/sr:.1f}s at {sr} Hz")
    except Exception as e:
        st.error(f"Error loading file: {e}")

elif audio_data is not None:
    try:
        # Streamlit audio_input returns bytes
        audio_to_process, sr = librosa.load(audio_data, sr=None)
        st.success(f"✓ Recording captured: {len(audio_to_process)/sr:.1f}s at {sr} Hz")
    except Exception as e:
        st.error(f"Error processing recording: {e}")

# --- Prediction ---
if audio_to_process is not None and sr is not None:
    st.subheader("🔍 Analysis")

    with st.spinner("Analyzing..."):
        pred, prob = predict_wheeze(audio_to_process, sr)

    if pred is not None:
        # Prediction result
        class_name = "🔴 WHEEZE DETECTED" if pred == 1 else "🟢 No Wheeze"
        wheeze_prob = prob[1] * 100
        no_wheeze_prob = prob[0] * 100

        st.markdown(f"## {class_name}")
        st.markdown(f"**Confidence:** {wheeze_prob:.1f}% wheeze, {no_wheeze_prob:.1f}% normal")

        # Visual gauge
        fig = go.Figure(data=[
            go.Bar(
                x=["No Wheeze", "Wheeze"],
                y=[no_wheeze_prob, wheeze_prob],
                marker=dict(color=["#2ecc71", "#e74c3c"]),
                text=[f"{no_wheeze_prob:.1f}%", f"{wheeze_prob:.1f}%"],
                textposition="auto",
            )
        ])
        fig.update_layout(
            title="Model Confidence",
            yaxis_title="Probability (%)",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_text = f"""
Timestamp: {datetime.now().isoformat()}
Prediction: {'WHEEZE' if pred == 1 else 'NO WHEEZE'}
Confidence: {wheeze_prob:.1f}%
Audio Duration: {len(audio_to_process)/sr:.2f}s
Sample Rate: {sr} Hz
"""
        try:
            result_file = RESULTS_DIR / f"test_{timestamp}.txt"
            result_file.write_text(result_text)
            st.success(f"✓ Result saved to: {result_file}")
        except Exception:
            # On a cloud server the disk may be read-only - that's fine,
            # the download button below still gives the user the result.
            st.info("Use the download button below to save this result.")

        # Download button
        st.download_button(
            label="📥 Download result",
            data=result_text,
            file_name=f"wheeze_test_{timestamp}.txt",
            mime="text/plain",
        )
    else:
        st.error("⚠️ Model not found. Make sure you've trained and saved the model first.")

st.divider()
st.markdown("""
---
### Tips for accurate testing:
- **Duration:** Aim for ~2 seconds of continuous breathing
- **Microphone:** Use a clean microphone or phone speaker
- **Background:** Minimize background noise
- **Format:** WAV format works best

### Next steps:
1. Test on 50–100 of your own patient recordings
2. Record accuracy statistics (how many wheeze vs no-wheeze)
3. Compare model predictions to your clinical assessment

---
*This is a proof-of-concept tool. Always validate against clinical expertise.*
""")
