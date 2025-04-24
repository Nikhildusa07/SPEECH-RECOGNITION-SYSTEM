# Speech-to-Text System

A basic speech-to-text system using `speechrecognition` and `Wav2Vec2` for transcribing short audio clips, with a web interface for uploading or recording audio.

## Features
- Transcribes audio using both SpeechRecognition (Google API) and Wav2Vec2.
- Hybrid approach with grammar scoring using TextBlob.
- Post-processing to correct common errors and remove unlikely phrases.
- Web interface (Flask) to upload audio files or record audio directly in the browser.
- Supports real-time recording via microphone in both CLI (`main.py`) and web interface (`app.py`).

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Add your audio files to the `audio_files/` folder.
6. Run the system:
   - CLI: `python src/main.py`
   - Web Interface: `python app.py` (then open `http://127.0.0.1:5000/` in your browser)

## Usage
- **CLI (`main.py`)**:
  - Run `python src/main.py` to transcribe an existing audio file (`audio_files/sample_audio.wav`) or record a new clip.
- **Web Interface (`app.py`)**:
  - Run `python app.py` and open `http://127.0.0.1:5000/`.
  - Upload a `.wav` file or record audio directly in the browser to transcribe.