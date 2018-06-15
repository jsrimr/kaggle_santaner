import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random

SR = 16000


class SpeechDataset(Dataset):
    def __init__(self, mode, label_to_int, wav_list, label_list=None):
        self.mode = mode
        self.label_to_int = label_to_int
        self.wav_list = wav_list
        self.label_list = label_list
        self.sr = SR
        self.n_silence = int(len(wav_list) * 0.1)

        # read all background noise here
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("/home1/irteam/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio/_background_noise_/*.wav")]

    def get_one_noise(self):
        """generates one single noise clip"""
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        # only takes out 16000
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    def __getitem__(self, idx):
        # reads one sample
        if idx < len(self.wav_list):
            wav_numpy = preprocess_mel(self.get_one_word_wav(idx))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}
            else:
                label = self.label_to_int.get(self.label_list[idx], len(self.label_to_int))
                return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}
        else:
            # generates silence here
            wav_numpy = preprocess_mel(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_to_int) + 1}

def preprocess_mel(data, n_mels=40):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram
