import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from src.utils import load_audio
import logging
import string
import transformers

transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = None
model = None

def load_wav2vec2_model():
    global processor, model
    if processor is None or model is None:
        logger.info("Loading Wav2Vec2 model and processor...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor, model

def transcribe_with_speechrecognition(audio_path, retries=2):
    recognizer = sr.Recognizer()
    for attempt in range(retries):
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                if not audio.frame_data:
                    logger.warning("SpeechRecognition: Empty audio file")
                    return "", 0.0
                result = recognizer.recognize_google(audio)  # Note: For production, set up a Google API key: recognizer.recognize_google(audio, key="YOUR_API_KEY")
                logger.info("SpeechRecognition result: %s", result)
                confidence = min(len(result.split()) * 0.1, 0.9) if result else 0.0
                return result, confidence
        except sr.UnknownValueError:
            logger.warning("SpeechRecognition: Could not understand audio (attempt %d/%d)", attempt + 1, retries)
        except sr.RequestError as e:
            logger.error("SpeechRecognition request error: %s", e)
        except Exception as e:
            logger.error("SpeechRecognition error: %s", e)
    return "", 0.0

def transcribe_with_wav2vec2(audio_path):
    try:
        processor, model = load_wav2vec2_model()
        input_values, sample_rate = load_audio(audio_path, pre_emphasis=0.0)
        if input_values.shape[1] == 0:
            logger.warning("Wav2Vec2: Empty audio input")
            return "", 0.0
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            result = processor.batch_decode(predicted_ids)[0].lower()
            result = result.translate(str.maketrans("", "", string.punctuation))
            logger.info("Wav2Vec2 result: %s", result)
            confidence = min(len(result.split()) * 0.05, 0.8) if result else 0.0
            return result, confidence
    except Exception as e:
        logger.error("Wav2Vec2 error: %s", e)
        return "", 0.0

def hybrid_transcription(audio_path):
    sr_result, sr_confidence = transcribe_with_speechrecognition(audio_path)
    wav2vec_result, wav2vec_confidence = transcribe_with_wav2vec2(audio_path)

    audio_duration = 0
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer = sr.Recognizer()
            audio_data = recognizer.record(source).get_wav_data()
            audio_duration = len(audio_data) / (16000 * 2)
    except Exception as e:
        logger.error("Error calculating audio duration: %s", e)

    if audio_duration < 5 and wav2vec_confidence >= sr_confidence and wav2vec_confidence > 0:
        final_result = wav2vec_result
        method = "Wav2Vec2 (short audio, higher or equal confidence)"
    elif sr_confidence > wav2vec_confidence and sr_result:
        final_result = sr_result
        method = "SpeechRecognition (higher confidence)"
    else:
        final_result = wav2vec_result if wav2vec_confidence > sr_confidence else sr_result
        method = "Fallback (highest confidence)"

    if not final_result:
        final_result = "Unable to transcribe audio. Please try again with a clearer recording."
        method = "Fallback (no transcription)"

    logger.info("Hybrid transcription: duration=%.2f s, method=%s, final=%s", audio_duration, method, final_result)

    return {
        "speech_recognition": sr_result,
        "speech_recognition_confidence": sr_confidence,
        "wav2vec2": wav2vec_result,
        "wav2vec2_confidence": wav2vec_confidence,
        "final_transcription": final_result,
        "method_used": method
    }