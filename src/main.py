import os
import speech_recognition as sr
from transcribe import hybrid_transcription

def record_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording... (Speak now)")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Recording stopped.")
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"Audio saved to: {audio_path}, size: {os.path.getsize(audio_path)} bytes")
        except sr.WaitTimeoutError:
            print("No audio detected within timeout. Please try again.")
            return False
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            return False
    return True

def main():
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio_files")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "cli_recorded_audio.wav")

    while True:
        choice = input("Do you want to record a new audio clip? (y/n): ").lower()
        if choice != 'y':
            print("Exiting.")
            break

        if record_audio(audio_path):
            try:
                result = hybrid_transcription(audio_path)
                print("\nTranscription Result:")
                print(f"SpeechRecognition: {result['speech_recognition']}")
                print(f"Wav2Vec2: {result['wav2vec2']}")
                print(f"Final Transcription: {result['final_transcription']}")
                print(f"Method Used: {result['method_used']}\n")
            except Exception as e:
                print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()