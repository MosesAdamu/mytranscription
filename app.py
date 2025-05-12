import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment

# Configuration
MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
st.title("🎙️ Robust Audio Transcriber")

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model(MODEL_SIZE)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def transcribe_audio(audio_path):
    try:
        model = load_whisper_model()
        result = model.transcribe(
            audio_path,
            fp16=False,  # Required for CPU compatibility
            verbose=True  # Shows basic progress in logs
        )
        return result["text"]
    except Exception as e:
        return f"Transcription failed: {str(e)}"

def main():
    uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("Transcribe"):
            with st.spinner("Processing..."):
                try:
                    # Create temporary file with proper extension
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        audio_path = tmp.name
                    
                    # Convert to WAV if needed
                    if file_extension.lower() != ".wav":
                        audio = AudioSegment.from_file(audio_path)
                        audio_path = audio_path + ".wav"
                        audio.export(audio_path, format="wav")
                    
                    # Transcribe
                    transcription = transcribe_audio(audio_path)
                    
                    # Display results
                    st.subheader("Transcription")
                    st.write(transcription)
                    
                    # Cleanup
                    os.unlink(audio_path)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if 'audio_path' in locals() and os.path.exists(audio_path):
                        os.unlink(audio_path)

if __name__ == "__main__":
    main()
