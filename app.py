import streamlit as st
from pydub import AudioSegment
import whisper
import tempfile
import os

# Set page title
st.title("Audio to Text Converter 🎤 (Whisper)")

# Instructions
st.markdown("""
1. Upload an audio file (MP3, WAV, etc.)
2. Click the 'Transcribe Audio' button
3. View your transcription (supports long files!)
""")

@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model to avoid reloading on every request"""
    return whisper.load_model("base")  # Use "tiny" for faster but less accurate results

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU compatibility
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a", "mp4"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=uploaded_file.type.split('/')[-1])
    
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio (this might take a while for large files)..."):
            try:
                # Save uploaded file temporarily
                audio_path = save_uploaded_file(uploaded_file)
                
                # Transcribe audio
                transcription = transcribe_audio(audio_path)
                
                # Clean up temporary file
                os.unlink(audio_path)
                
                # Display results
                st.subheader("Transcription Result")
                st.write(transcription)
                
                # Add download button
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

# Add model size recommendation
st.markdown("---")
st.info("""
**Tips:**
- For faster results: Use 'tiny' model in `load_whisper_model()`
- For better accuracy: Use 'small' or 'medium' models (requires more memory)
- Max recommended file size: 25MB (Streamlit Cloud limit)
""")
