import os
import speech_recognition as sr
from src.transcribe import hybrid_transcription

def record_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording... (Speak now)")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Recording stopped.")
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return False
        except Exception as e:
            print(f"Recording error: {e}")
            return False
    return True

def main():
    audio_dir = "audio_files"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "cli_recorded_audio.wav")

    while True:
        choice = input("Record a new audio clip? (y/n): ").lower()
        if choice != 'y':
            break
        if record_audio(audio_path):
            result = hybrid_transcription(audio_path)
            print("\nTranscription Result:\n")
            print(result["final_transcription"])

if __name__ == "__main__":
    main()
