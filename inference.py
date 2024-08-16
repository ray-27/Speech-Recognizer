import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

class SpeechRecognitionInference:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = SpeechTransformer(num_features=128, num_classes=29).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Assume the model was trained with Mel Spectrogram as input
        self.transform = MelSpectrogram(
            sample_rate=16000,  # Adjust as per the training configuration
            n_fft=400,
            hop_length=160,
            n_mels=128
        )

    def preprocess_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        mel_spec = self.transform(waveform).squeeze(0).transpose(0, 1)  # Shape to (time, features)
        return mel_spec.to(self
