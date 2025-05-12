from flask import Flask, request, render_template, send_from_directory
import os
from src.transcribe import hybrid_transcription
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_FOLDER = 'audio_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        filename = "uploaded_audio.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    elif 'audio_data' in request.files:
        audio_data = request.files['audio_data']
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.webm")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")
        try:
            audio_data.save(temp_file_path)
            audio = AudioSegment.from_file(temp_file_path, format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(file_path, format="wav")
            os.remove(temp_file_path)
        except Exception as e:
            return f"Audio processing error: {e}", 500
    else:
        return "No audio data provided", 400

    try:
        result = hybrid_transcription(file_path)
    except Exception as e:
        return f"Transcription failed: {e}", 500

    return render_template("result.html", result=result, audio_filename=os.path.basename(file_path))

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
