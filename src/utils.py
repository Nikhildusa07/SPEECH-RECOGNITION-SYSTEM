import logging
import numpy as np
from noisereduce import reduce_noise
from pydub import AudioSegment
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(audio_path, pre_emphasis=0.0):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        rate = 16000
        audio_samples = reduce_noise(y=audio_samples, sr=rate, prop_decrease=0.8)
        if pre_emphasis > 0:
            audio_samples = np.append(audio_samples[0], audio_samples[1:] - pre_emphasis * audio_samples[:-1])
        max_val = np.max(np.abs(audio_samples))
        if max_val > 0:
            audio_samples = audio_samples / max_val
        else:
            audio_samples = audio_samples / (max_val + 1e-6)
        audio_samples = audio_samples.astype(np.float32)
        logger.info("Loaded audio: path=%s, sample_rate=%d, length=%d", audio_path, rate, len(audio_samples))
        return torch.tensor(audio_samples, dtype=torch.float32).unsqueeze(0), rate
    except Exception as e:
        logger.error("Error loading audio %s: %s", audio_path, e)
        return torch.tensor([], dtype=torch.float32).unsqueeze(0), 16000