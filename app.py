import streamlit as st
import whisper
import tempfile
import os
import subprocess

# Verify FFmpeg installation
try:
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
except Exception as e:
    st.error("""
    FFmpeg not found! This app requires FFmpeg to process audio files.
    Please contact the app administrator to ensure FFmpeg is installed.
    """)
    st.stop()

st.title("Audio to Text Converter 🎤 (Whisper + FFmpeg)")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def transcribe_audio(audio_path):
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path, fp16=False)
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "mp4", "m4a", "ogg"])

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("Transcribe Audio"):
        with st.spinner("Processing..."):
            try:
                audio_path = save_uploaded_file(uploaded_file)
                transcription = transcribe_audio(audio_path)
                os.unlink(audio_path)
                
                st.subheader("Result")
                st.write(transcription)
                
                st.download_button(
                    "Download Transcription",
                    transcription,
                    "transcription.txt"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
