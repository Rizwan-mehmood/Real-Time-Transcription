import streamlit as st
import pyaudio
import torch
import whisper
import librosa
import threading
import queue
import numpy as np
import time
import logging

# ---------------------------
# Logging Configuration
# ---------------------------
# Configure a separate logger for background threads to avoid Streamlit context issues
thread_logger = logging.getLogger("background_threads")
thread_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
thread_logger.addHandler(handler)
thread_logger.propagate = False  # Prevent propagation to root logger

# Configure main logger for Streamlit components
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------
# Load the Whisper Model
# ---------------------------
thread_logger.info("Loading Whisper model...")
model = whisper.load_model("base")
thread_logger.info("Model loaded successfully.")

# ---------------------------
# Audio Configuration
# ---------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 2  # Use audio device index 2 as requested
DURATION_PER_CHUNK = 5  # seconds of audio per chunk

# Initialize PyAudio and a queue for audio chunks
p = pyaudio.PyAudio()
audio_queue = queue.Queue()

# Global list to hold transcriptions and a lock for thread safety
transcriptions = []
transcription_lock = threading.Lock()

# Global thread references
record_thread = None
process_thread = None


# ---------------------------
# Background Thread Functions
# ---------------------------
def record_audio():
    """Record audio from the microphone and enqueue resampled audio chunks."""
    thread_logger.info("Starting audio recording on device index %d", DEVICE_INDEX)
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=DEVICE_INDEX,
            frames_per_buffer=CHUNK,
            start=False,
        )
    except Exception as e:
        thread_logger.error("Error opening audio stream: %s", e)
        return

    stream.start_stream()
    thread_logger.info("Audio stream started. Listening for audio...")
    bytes_per_sample = p.get_sample_size(FORMAT)
    required_bytes = RATE * DURATION_PER_CHUNK * bytes_per_sample * CHANNELS

    while st.session_state.get("is_running", True):
        frames = []
        bytes_read = 0

        while bytes_read < required_bytes and st.session_state.get("is_running", True):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                thread_logger.error("Error reading audio data: %s", e)
                continue
            frames.append(data)
            bytes_read += len(data)

        if not st.session_state.get("is_running", True):
            break

        audio_data = b"".join(frames)
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        audio_resampled = librosa.resample(audio_np, orig_sr=RATE, target_sr=16000)
        audio_queue.put(audio_resampled)
        thread_logger.info("Audio chunk recorded and queued for transcription.")

    stream.stop_stream()
    stream.close()
    thread_logger.info("Audio recording stopped.")


def process_audio():
    """Transcribe audio chunks from the queue and update the transcription list."""
    while st.session_state.get("is_running", True):
        try:
            audio_np = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            thread_logger.info("Starting transcription for an audio chunk...")
            result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
            transcription = result["text"].strip()
            thread_logger.info("Transcription complete: %s", transcription)

            with transcription_lock:
                transcriptions.append(transcription)

            with open("transcriptions.txt", "a", encoding="utf-8") as f:
                f.write(f"{transcription}\n")
        except Exception as e:
            thread_logger.error("Error transcribing audio: %s", e)
        finally:
            audio_queue.task_done()
    thread_logger.info("Audio processing stopped.")


def start_background_threads():
    """Start background threads for recording and processing if they are not alive."""
    global record_thread, process_thread
    if record_thread is None or not record_thread.is_alive():
        record_thread = threading.Thread(target=record_audio, daemon=True)
        record_thread.start()
    if process_thread is None or not process_thread.is_alive():
        process_thread = threading.Thread(target=process_audio, daemon=True)
        process_thread.start()


# ---------------------------
# Streamlit User Interface
# ---------------------------
st.title("Real-Time Transcription with Start/Stop & Download")
st.write("Transcription is performed using the Whisper model in the background.")

if "is_running" not in st.session_state:
    st.session_state["is_running"] = True

col1, col2 = st.columns(2)
with col1:
    if st.session_state["is_running"]:
        if st.button("Stop Transcription"):
            st.session_state["is_running"] = False
            st.success("Transcription stopped.")
    else:
        if st.button("Start Transcription"):
            st.session_state["is_running"] = True
            start_background_threads()
            st.success("Transcription started.")

with col2:
    if st.button("Clear Transcriptions"):
        with transcription_lock:
            transcriptions.clear()
        st.success("Transcriptions cleared.")

with transcription_lock:
    transcript_text = "\n\n".join(transcriptions)
st.download_button(
    label="Download Transcriptions",
    data=transcript_text,
    file_name="transcriptions.txt",
    mime="text/plain",
)

if st.session_state["is_running"]:
    start_background_threads()

transcription_placeholder = st.empty()

while True:
    with transcription_lock:
        text_to_display = " ".join(transcriptions)
    transcription_placeholder.text(text_to_display)
    time.sleep(1)
