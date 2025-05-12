import streamlit as st
import whisper
import tempfile
import os
import numpy as np
from pydub import AudioSegment
import torch

# Configuration
MODEL_SIZE = "tiny.en"  # Use English-specific model for better compatibility
SAMPLE_RATE = 16000      # Whisper requires 16kHz audio

st.title("🎙️ Reliable Audio Transcription")

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model(MODEL_SIZE)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def convert_to_whisper_format(audio_path):
    """Convert audio to Whisper's expected format with proper tensor shaping"""
    audio = AudioSegment.from_file(audio_path)
    
    # Ensure mono channel and correct sample rate
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize and convert to float32
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        
    return samples

def transcribe_audio(audio_path):
    try:
        model = load_whisper_model()
        audio = convert_to_whisper_format(audio_path)
        result = model.transcribe(audio.reshape(1, -1),  # Add batch dimension
                    fp16=torch.cuda.is_available())      # Use GPU if available
        return result["text"]
    except Exception as e:
        return f"Transcription failed: {str(e)}"

def main():
    uploaded_file = st.file_uploader("Upload audio (max 25MB)", 
                                   type=["mp3", "wav", "m4a"],
                                   accept_multiple_files=False)
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("Transcribe"):
            with st.spinner("Processing audio..."):
                try:
                    # Save uploaded file with proper extension
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        audio_path = tmp.name
                    
                    # Process and transcribe
                    transcription = transcribe_audio(audio_path)
                    
                    # Cleanup
                    os.unlink(audio_path)
                    
                    # Display results
                    st.subheader("Transcription")
                    st.write(transcription)
                    
                    # Download button
                    st.download_button(
                        "Download Transcription",
                        transcription,
                        file_name="transcription.txt"
                    )
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)

if __name__ == "__main__":
    main()
