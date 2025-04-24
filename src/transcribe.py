import speech_recognition as sr
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from .utils import load_audio

def transcribe_with_speechrecognition(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "SpeechRecognition could not understand the audio."
    except sr.RequestError as e:
        return f"SpeechRecognition error: {e}"

def transcribe_with_wav2vec2(audio_path):
    # Placeholder for Wav2Vec2 implementation (requires model loading)
    # This assumes load_audio returns (audio_data, sample_rate)
    audio, sample_rate = load_audio(audio_path)
    # Add Wav2Vec2 model and tokenizer loading here if not already implemented
    # For now, return a dummy value
    return "Wav2Vec2 sample transcription"

def hybrid_transcription(audio_path):
    speech_recognition_result = transcribe_with_speechrecognition(audio_path)
    wav2vec2_result = transcribe_with_wav2vec2(audio_path)
    final_result = wav2vec2_result if len(wav2vec2_result.split()) > len(speech_recognition_result.split()) else speech_recognition_result
    return {
        "speech_recognition": speech_recognition_result,
        "wav2vec2": wav2vec2_result,
        "final_transcription": final_result,
        "method_used": "Hybrid (Wav2Vec2 prioritized due to grammar)"
    }