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
from streamlit_autorefresh import st_autorefresh
from pytube import YouTube
import tempfile
import os
import yt_dlp

# Import helper to attach threads to the current Streamlit script context.
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.markdown(
    "<h1 style='font-size: 40px;'>VoiceVerse</h1>", unsafe_allow_html=True
)
nav_option = st.sidebar.radio(
    "Go to",
    [
        "Real Time Audio to Text Conversion",
        "Youtube Link to Text",
        "Audio/Video (Upload) to Text",
        "Text to Audio Conversion",
    ],
)

# ---------------------------
# Real Time Audio to Text Conversion Tab
# ---------------------------
if nav_option == "Real Time Audio to Text Conversion":
    st.title("Real-Time Transcription with Whisper AI")
    st.write(
        "This interface captures audio from the output device (index 2) and transcribes using the Whisper base model."
    )

    # ---------------------------
    # Logging Configuration (once)
    # ---------------------------
    if "logger_configured" not in st.session_state:
        thread_logger = logging.getLogger("background_threads")
        thread_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        thread_logger.addHandler(handler)
        thread_logger.propagate = False
        st.session_state["logger_configured"] = True
    else:
        thread_logger = logging.getLogger("background_threads")

    # ---------------------------
    # Load the Whisper Model (once)
    # ---------------------------
    if "whisper_model" not in st.session_state:
        st.info("Loading Whisper model... please wait")
        model = whisper.load_model("base")
        st.session_state["whisper_model"] = model
        thread_logger.info("Model loaded successfully.")
    else:
        model = st.session_state["whisper_model"]

    # ---------------------------
    # Audio Configuration
    # ---------------------------
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    DEVICE_INDEX = 2  # use the output device at index 2
    DURATION_PER_CHUNK = 5  # seconds of audio per chunk

    if "pyaudio_instance" not in st.session_state:
        st.session_state["pyaudio_instance"] = pyaudio.PyAudio()
    if "audio_queue" not in st.session_state:
        st.session_state["audio_queue"] = queue.Queue()

    # Global transcription list and running flag stored in session state
    if "transcriptions" not in st.session_state:
        st.session_state["transcriptions"] = []
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False

    # ---------------------------
    # Background Thread Functions
    # ---------------------------
    def record_audio():
        """Record audio from the selected device and queue resampled chunks."""
        thread_logger.info("Starting audio recording on device index %d", DEVICE_INDEX)
        p_inst = st.session_state["pyaudio_instance"]
        try:
            stream = p_inst.open(
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
        bytes_per_sample = p_inst.get_sample_size(FORMAT)
        required_bytes = RATE * DURATION_PER_CHUNK * bytes_per_sample * CHANNELS

        while st.session_state.get("is_running", False):
            frames = []
            bytes_read = 0

            while bytes_read < required_bytes and st.session_state.get(
                "is_running", False
            ):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception as e:
                    thread_logger.error("Error reading audio data: %s", e)
                    continue
                frames.append(data)
                bytes_read += len(data)

            if not st.session_state.get("is_running", False):
                break

            audio_data = b"".join(frames)
            # Convert to numpy array and normalize
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            # Resample to 16000 Hz as required by Whisper
            audio_resampled = librosa.resample(audio_np, orig_sr=RATE, target_sr=16000)
            st.session_state["audio_queue"].put(audio_resampled)
            thread_logger.info("Audio chunk recorded and queued for transcription.")

        stream.stop_stream()
        stream.close()
        thread_logger.info("Audio recording stopped.")

    def process_audio():
        """Transcribe queued audio chunks and update the transcription list."""
        while st.session_state.get("is_running", False):
            try:
                audio_np = st.session_state["audio_queue"].get(timeout=1)
            except queue.Empty:
                continue

            try:
                thread_logger.info("Starting transcription for an audio chunk...")
                result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
                transcription = result["text"].strip()
                thread_logger.info("Transcription complete: %s", transcription)
                st.session_state["transcriptions"].append(transcription)
            except Exception as e:
                thread_logger.error("Error transcribing audio: %s", e)
            finally:
                st.session_state["audio_queue"].task_done()
        thread_logger.info("Audio processing stopped.")

    def start_background_threads():
        """Start recording and processing threads if not already running."""
        if (
            "record_thread" not in st.session_state
            or not st.session_state["record_thread"].is_alive()
        ):
            record_thread = threading.Thread(target=record_audio, daemon=True)
            add_script_run_ctx(record_thread)
            record_thread.start()
            st.session_state["record_thread"] = record_thread
        if (
            "process_thread" not in st.session_state
            or not st.session_state["process_thread"].is_alive()
        ):
            process_thread = threading.Thread(target=process_audio, daemon=True)
            add_script_run_ctx(process_thread)
            process_thread.start()
            st.session_state["process_thread"] = process_thread

    # ---------------------------
    # UI Controls Layout
    # ---------------------------
    cols = st.columns(3)
    with cols[0]:
        if st.session_state["is_running"]:
            if st.button("Pause Transcription"):
                st.session_state["is_running"] = False
                st.success("Transcription paused.")
        else:
            if st.button("Start Transcription"):
                st.session_state["is_running"] = True
                start_background_threads()
                st.success("Transcription started.")

    with cols[1]:
        if st.button("Clear Transcriptions"):
            st.session_state["transcriptions"] = []
            st.success("Transcriptions cleared.")

    with cols[2]:
        format_option = st.selectbox("Download format", ["txt", "srt"])

    # ---------------------------
    # Download Button
    # ---------------------------
    transcript_text = " ".join(st.session_state["transcriptions"])
    file_name = f"transcriptions.{format_option}"
    st.download_button(
        label="Download Transcriptions",
        data=transcript_text,
        file_name=file_name,
        mime="text/plain",
    )

    # ---------------------------
    # Auto-Refresh UI Every Second
    # ---------------------------
    _ = st_autorefresh(interval=1000, limit=10000, key="transcription_autorefresh")

    # ---------------------------
    # Display Live Transcriptions
    # ---------------------------
    st.markdown("### Live Transcription:")
    st.text_area("Live Transcription", transcript_text, height=300)

# ---------------------------
# Youtube Link to Text Tab
# ---------------------------
elif nav_option == "Youtube Link to Text":
    st.header("Youtube Link to Text")

    # Set up a dedicated logger for this tab
    yt_logger = logging.getLogger("youtube_tab")
    yt_logger.setLevel(logging.INFO)
    if not yt_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        yt_logger.addHandler(handler)

    # Input for YouTube URL
    youtube_url = st.text_input(
        "Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=..."
    )

    # Option to choose conversion type
    conversion_option = st.radio(
        "Select conversion type", ("Generate Audio", "Generate Text")
    )

    if st.button("Generate", key="yt_generate"):
        if not youtube_url:
            st.error("Please enter a YouTube URL.")
        else:
            try:
                # Create and update progress bar
                progress_bar = st.progress(0)
                st.write("Processing your video...")
                yt_logger.info("Starting processing for URL: %s", youtube_url)
                progress_bar.progress(10)
                yt_logger.info(
                    "Downloading video stream using yt_dlp for URL: %s", youtube_url
                )
                progress_bar.progress(30)

                import yt_dlp
                import tempfile, os

                temp_dir = tempfile.mkdtemp()
                yt_logger.info("Temporary directory created at: %s", temp_dir)
                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                    "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
                    "http_headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/90.0.4430.93 Safari/537.36"
                    },
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(youtube_url, download=True)
                    video_path = ydl.prepare_filename(info_dict)
                yt_logger.info("Video downloaded to: %s", video_path)
                progress_bar.progress(60)

                # Extract audio from the downloaded video using ffmpeg-python
                try:
                    try:
                        import ffmpeg
                    except ImportError as imp_err:
                        st.error(
                            "ffmpeg-python is not installed. Please install it using: pip install ffmpeg-python"
                        )
                        yt_logger.error("ffmpeg-python import failed: %s", imp_err)
                        raise
                    yt_logger.info("Extracting audio from video using ffmpeg-python.")
                    audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
                    ffmpeg.input(video_path).output(
                        audio_path, acodec="mp3", audio_bitrate="128k"
                    ).run(overwrite_output=True, quiet=True)
                    yt_logger.info("Audio extracted to: %s", audio_path)
                    progress_bar.progress(70)
                except Exception as e:
                    yt_logger.error("Audio extraction failed: %s", e)
                    st.error(f"Audio extraction failed: {e}")
                    raise

                if conversion_option == "Generate Audio":
                    # Read audio file bytes for st.audio and download button.
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        "Download Audio",
                        data=audio_bytes,
                        file_name="audio.mp3",
                        mime="audio/mp3",
                    )
                    yt_logger.info("Audio generated and ready for download.")
                    progress_bar.progress(100)

                elif conversion_option == "Generate Text":
                    # Ensure the Whisper model is loaded (reuse if already loaded)
                    if "whisper_model" not in st.session_state:
                        with st.spinner("Loading Whisper model..."):
                            model = whisper.load_model("base")
                            st.session_state["whisper_model"] = model
                            yt_logger.info("Whisper model loaded for transcription.")
                    else:
                        model = st.session_state["whisper_model"]
                    progress_bar.progress(80)
                    with st.spinner("Transcribing..."):
                        result = model.transcribe(
                            audio_path, fp16=torch.cuda.is_available()
                        )
                        transcription_text = result["text"]
                    st.text_area("Transcription", transcription_text, height=300)

                    # Option to choose download format for transcription
                    text_format = st.selectbox(
                        "Select download text format", ["txt", "srt"]
                    )
                    file_name = f"transcription.{text_format}"
                    st.download_button(
                        "Download Transcription",
                        data=transcription_text,
                        file_name=file_name,
                        mime="text/plain",
                    )
                    yt_logger.info("Transcription generated and ready for download.")
                    progress_bar.progress(100)
            except Exception as e:
                yt_logger.error("Error processing YouTube video: %s", e)
                st.error(f"Error processing YouTube video: {e}")


elif nav_option == "Audio/Video (Upload) to Text":
    st.header("Audio/Video (Upload) to Text")
    st.write(
        "Upload an audio or video file and click 'Generate' to convert it to text."
    )

    # Allow user to upload an audio or video file
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["mp3", "wav", "m4a", "aac", "mp4", "mov", "avi", "mkv"],
    )

    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        # Get the file extension to determine if it's video or audio
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if st.button("Generate", key="av_generate"):
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                progress_bar.progress(5)
                st.info("Processing file...")

                # Save uploaded file to a temporary file
                import tempfile, os, ffmpeg

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix="." + file_ext
                ) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                progress_bar.progress(15)

                # Determine if the file is a video file and extract audio if needed
                video_ext = ["mp4", "mov", "avi", "mkv"]
                if file_ext in video_ext:
                    st.write("Video file detected. Extracting audio...")
                    progress_bar.progress(25)
                    # Create a temporary audio file path
                    audio_temp_path = temp_file_path + "_audio.mp3"
                    # Use ffmpeg-python to extract audio from the video file
                    ffmpeg.input(temp_file_path).output(
                        audio_temp_path, acodec="mp3", audio_bitrate="128k"
                    ).run(overwrite_output=True, quiet=True)
                    st.write("Audio extracted.")
                    progress_bar.progress(40)
                    file_to_transcribe = audio_temp_path
                else:
                    file_to_transcribe = temp_file_path
                    progress_bar.progress(40)

                # Load Whisper model (reuse if already loaded)
                if "whisper_model" not in st.session_state:
                    with st.spinner("Loading Whisper model..."):
                        model = whisper.load_model("base")
                        st.session_state["whisper_model"] = model
                else:
                    model = st.session_state["whisper_model"]
                progress_bar.progress(50)

                # Transcribe the file using Whisper
                with st.spinner("Transcribing..."):
                    result = model.transcribe(
                        file_to_transcribe, fp16=torch.cuda.is_available()
                    )
                    transcription_text = result["text"]
                progress_bar.progress(90)

                # Display the transcription in a text area
                st.text_area("Transcription", transcription_text, height=300)

                # Let the user choose the download format (TXT or SRT)
                text_format = st.selectbox(
                    "Select download text format", ["txt", "srt"]
                )
                file_name = f"transcription.{text_format}"
                st.download_button(
                    "Download Transcription",
                    data=transcription_text,
                    file_name=file_name,
                    mime="text/plain",
                )
                progress_bar.progress(100)
            except Exception as e:
                st.error(f"Error processing file: {e}")


elif nav_option == "Text to Audio Conversion":
    st.header("Text to Audio Conversion")
    st.write("Enter text below and click 'Generate Audio' to convert it to speech.")

    # Text input area for user to enter text
    text_input = st.text_area("Enter text here:", height=300)

    if st.button("Generate Audio", key="generate_audio"):
        if not text_input.strip():
            st.error("Please enter some text to convert!")
        else:
            # Initialize a progress bar
            progress_bar = st.progress(0)
            progress_bar.progress(5)

            # Detect language using langdetect
            with st.spinner("Detecting language..."):
                try:
                    from langdetect import detect

                    detected_lang = detect(text_input)
                    st.write("Detected language:", detected_lang)
                except Exception as e:
                    st.error("Language detection failed: " + str(e))
                    detected_lang = "en"  # fallback to English
            progress_bar.progress(30)

            # Generate audio using gTTS
            with st.spinner("Generating audio..."):
                try:
                    from gtts import gTTS

                    tts = gTTS(text_input, lang=detected_lang)
                    import tempfile

                    # Save audio to a temporary file
                    temp_audio_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp3"
                    )
                    tts.save(temp_audio_file.name)
                    audio_file_path = temp_audio_file.name
                except Exception as e:
                    st.error("Audio generation failed: " + str(e))
                    audio_file_path = None
            progress_bar.progress(60)

            if audio_file_path:
                st.success("Audio generated successfully!")
                # Read the audio file bytes
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                # Display an audio player
                st.audio(audio_bytes, format="audio/mp3")
                # Provide a download button
                st.download_button(
                    "Download Audio",
                    data=audio_bytes,
                    file_name="generated_audio.mp3",
                    mime="audio/mp3",
                )
            progress_bar.progress(100)
