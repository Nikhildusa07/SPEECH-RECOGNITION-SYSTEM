import librosa
import torch

def load_audio(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    return torch.tensor(audio).unsqueeze(0), rate
