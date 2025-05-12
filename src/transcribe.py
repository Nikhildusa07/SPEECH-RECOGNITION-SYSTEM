import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from .utils import load_audio

# Load model and processor once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_with_speechrecognition(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return ""

def transcribe_with_wav2vec2(audio_path):
    input_values, sample_rate = load_audio(audio_path)
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].lower()

def hybrid_transcription(audio_path):
    sr_result = transcribe_with_speechrecognition(audio_path)
    wav2vec_result = transcribe_with_wav2vec2(audio_path)
    final_result = wav2vec_result if len(wav2vec_result) > len(sr_result) else sr_result
    return {
        "speech_recognition": sr_result,
        "wav2vec2": wav2vec_result,
        "final_transcription": final_result,
        "method_used": "Hybrid (best of both)"
    }
