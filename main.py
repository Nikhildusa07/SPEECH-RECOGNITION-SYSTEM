import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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
            return True
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return False
        except Exception as e:
            print(f"Recording error: {e}")
            return False

def main():
    audio_dir = "audio_files"
    os.makedirs(audio_dir, exist_ok=True)

    while True:
        choice = input("Record a new audio clip? (y/n): ").lower()
        if choice != 'y':
            break
        audio_path = os.path.join(audio_dir, f"cli_recorded_audio_{int(time.time())}.wav")
        if record_audio(audio_path):
            try:
                result = hybrid_transcription(audio_path)
                print("\nTranscription Result:\n")
                print(result["final_transcription"])
                if "Unable to transcribe" in result["final_transcription"]:
                    retry = input("Transcription failed. Retry? (y/n): ").lower()
                    if retry != 'y':
                        break
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"Cleaned up file: {audio_path}")

if __name__ == "__main__":
    main()