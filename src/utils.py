import librosa

def load_audio(audio_path):
    print(f"Loading audio from: {audio_path}")
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    print(f"Loaded audio shape: {audio.shape}, sample_rate: {sample_rate}")
    return audio, sample_rate