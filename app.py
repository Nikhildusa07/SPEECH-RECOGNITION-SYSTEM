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
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("Received request to /transcribe")
    if 'file' in request.files:
        print("Handling uploaded file")
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return "No file selected", 400
        filename = "uploaded_audio.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving uploaded file to: {file_path}")
        file.save(file_path)
    elif 'audio_data' in request.files:
        print("Handling recorded audio")
        audio_data = request.files['audio_data']
        temp_filename = "temp_recorded_audio.webm"
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        filename = "recorded_audio.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            audio_data.save(temp_file_path)
            file_size = os.path.getsize(temp_file_path)
            print(f"Temporary recorded audio saved, size: {file_size} bytes")
            if file_size == 0:
                raise ValueError("Temporary file is empty")
        except Exception as e:
            print(f"Error saving temporary recorded audio: {str(e)}")
            return f"Error saving temporary recorded audio: {str(e)}", 500

        try:
            audio = AudioSegment.from_file(temp_file_path, format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(file_path, format="wav")
            print(f"Converted audio to PCM WAV and saved to: {file_path}, size: {os.path.getsize(file_path)} bytes")
        except Exception as e:
            print(f"Error converting audio to WAV: {str(e)}")
            return f"Error converting audio to WAV: {str(e)}", 500
        finally:
            try:
                os.remove(temp_file_path)
                print("Temporary file removed")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
    else:
        print("No audio data provided")
        return "No audio data provided", 400
    
    print("Starting transcription")
    try:
        result = hybrid_transcription(file_path)
        print("Transcription completed:", result)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Error during transcription: {str(e)}", 500

    return render_template('result.html', result=result, audio_filename=filename)

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    print(f"Serving audio file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)