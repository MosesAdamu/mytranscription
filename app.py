import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

# Set page title
st.title("Audio to Text Converter 🎤")

# Instructions
st.markdown("""
1. Upload an audio file (WAV, MP3, etc.)
2. Click the 'Transcribe Audio' button
3. Wait for the transcription to complete
4. View results and download as text file
""")

# Initialize recognizer
recognizer = sr.Recognizer()

def save_uploaded_file(uploaded_file, suffix):
    """Save uploaded file to a temporary location with the given suffix"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def convert_to_wav(audio_file_path, original_suffix):
    """Convert any audio file to WAV format with 16kHz sampling rate and mono channel"""
    audio = AudioSegment.from_file(audio_file_path, format=original_suffix.lstrip('.'))
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = audio_file_path + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path):
    """Transcribe audio using Google Speech Recognition"""
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=uploaded_file.type)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            file_suffix = os.path.splitext(uploaded_file.name)[1]
            original_path = save_uploaded_file(uploaded_file, file_suffix)
            
            # Convert to WAV if necessary
            if file_suffix.lower() != ".wav":
                audio_path = convert_to_wav(original_path, file_suffix)
                os.unlink(original_path)  # Remove original file
            else:
                audio_path = original_path
            
            # Transcribe audio
            transcription = transcribe_audio(audio_path)
            
            # Clean up temporary files
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
