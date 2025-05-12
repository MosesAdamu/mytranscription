import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment
import time

# Configuration
MODEL_SIZE = "tiny"  # Faster processing (options: tiny, base, small, medium, large)
MAX_FILE_DURATION = 10 * 60  # 10 minutes in seconds

st.title("⚡ Fast Audio to Text Converter")

# Progress bar callback for Whisper
def progress_callback(progress):
    bar.progress(min(int(progress * 100), 100))

@st.cache_resource
def load_whisper_model():
    st.info("⚠️ First-time setup: Downloading model... (this may take a few minutes)")
    return whisper.load_model(MODEL_SIZE)

def optimize_audio(input_path):
    """Convert audio to optimal format for faster processing"""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    optimized_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    audio.export(optimized_path, format="wav", codec="pcm_s16le")
    return optimized_path

def estimate_processing_time(duration):
    """Heuristic time estimation based on model size"""
    return duration * {"tiny": 0.3, "base": 0.5, "small": 1}.get(MODEL_SIZE, 1.5)

def transcribe_audio(audio_path):
    model = load_whisper_model()
    global bar
    bar = st.progress(0)
    
    start_time = time.time()
    result = model.transcribe(
        audio_path,
        fp16=False,
        verbose=False,  # Disable console output
        task="transcribe",
        progress_callback=progress_callback
    )
    
    processing_time = time.time() - start_time
    st.session_state.last_processing_time = processing_time
    return result["text"]

# File uploader with size warning
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"],
                                help=f"Max recommended duration: {MAX_FILE_DURATION//60} minutes")

if uploaded_file:
    audio_path = None
    try:
        # Save and optimize audio
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        # Get audio duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000  # Convert to seconds
        
        if duration > MAX_FILE_DURATION:
            st.warning(f"⚠️ Long audio ({duration//60}min). Processing may take time!")
        
        st.audio(uploaded_file)
        st.caption(f"Detected: {duration:.1f}s audio | Estimated processing time: ~{estimate_processing_time(duration):.1f}s")

        if st.button("🚀 Transcribe Audio"):
            optimized_path = optimize_audio(audio_path)
            
            with st.spinner("Optimizing audio format..."):
                optimized_path = optimize_audio(audio_path)
            
            with st.status("Transcribing...", expanded=True) as status:
                st.write("🔧 Initializing transcription engine")
                transcription = transcribe_audio(optimized_path)
                status.update(label="Transcription complete!", state="complete")
            
            st.subheader("Result")
            st.write(transcription)
            
            st.download_button(
                "📥 Download Transcription",
                transcription,
                "transcription.txt"
            )
            
            if "last_processing_time" in st.session_state:
                st.caption(f"Processing time: {st.session_state.last_processing_time:.1f} seconds")

    finally:
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

st.markdown("---")
st.info(f"""
**Performance Tips:**
1. Current model: `{MODEL_SIZE}` (change code to 'base' for better accuracy)
2. Keep files under 5 minutes for fastest results
3. Use WAV format for optimal processing speed
""")
