import sys
import os
import time
from flask import Flask, request, render_template, send_from_directory, session
from flask_session import Session
import logging
from src.transcribe import hybrid_transcription
from pydub import AudioSegment

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

UPLOAD_FOLDER = 'audio_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def cleanup_old_files():
    if 'audio_filename' in session:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], session['audio_filename'])
        if os.path.exists(audio_path) and time.time() - os.path.getmtime(audio_path) > 3600:  # 1 hour
            os.remove(audio_path)
            logger.info("Cleaned up old file: %s", audio_path)
            session.pop('audio_filename', None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file_path = None
    audio_filename = None
    result = {
        "speech_recognition": "",
        "speech_recognition_confidence": 0.0,
        "wav2vec2": "",
        "wav2vec2_confidence": 0.0,
        "final_transcription": "Unable to transcribe audio. Please try again with a clearer recording.",
        "method_used": "None"
    }

    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.warning("No file selected")
                result["final_transcription"] = "No file selected. Please upload an audio file."
            elif not allowed_file(file.filename):
                logger.warning("Invalid file format: %s", file.filename)
                result["final_transcription"] = "Invalid file format. Please upload a .wav, .mp3, or .webm file."
            else:
                filename = f"uploaded_audio_{int(time.time())}.wav"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                audio_filename = os.path.basename(file_path)
                session['audio_filename'] = audio_filename
                logger.info("File uploaded: %s", file_path)

        elif 'audio_data' in request.files:
            audio_data = request.files['audio_data']
            if audio_data.filename == '':
                logger.warning("No audio data provided")
                result["final_transcription"] = "No audio data provided. Please record audio."
            else:
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{int(time.time())}.webm")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"recorded_audio_{int(time.time())}.wav")
                audio_data.save(temp_file_path)
                try:
                    audio = AudioSegment.from_file(temp_file_path, format="webm")
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(file_path, format="wav")
                    audio_filename = os.path.basename(file_path)
                    session['audio_filename'] = audio_filename
                    logger.info("Audio recorded and converted: %s", file_path)
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        else:
            logger.warning("No audio data provided")
            result["final_transcription"] = "No audio data provided. Please upload or record audio."

        if file_path:
            try:
                transcription_result = hybrid_transcription(file_path)
                result.update(transcription_result)
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                result["final_transcription"] = "Transcription failed. Please try again with a clearer recording."

    except Exception as e:
        logger.error("Audio processing error: %s", e)
        result["final_transcription"] = "An error occurred. Please try again."

    return render_template("result.html", result=result, audio_filename=audio_filename if audio_filename else "unknown.wav")

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(audio_path):
        logger.warning("Audio file not found: %s", filename)
        return "Audio file not found", 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    original = data.get('original', '')
    corrected = data.get('corrected', '')
    logger.info("Feedback received: original=%s, corrected=%s", original, corrected)
    with open('feedback.txt', 'a') as f:
        f.write(f"Original: {original}, Corrected: {corrected}\n")
    return {"status": "success"}

if __name__ == '__main__':
    app.run(debug=True)