import numpy as np
from typing import Dict
import torchaudio
import torch
import pathlib
import os


class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


from .basic_pitch_torch.model import BasicPitchTorch
from .basic_pitch_torch.inference import predict
import torch


class BasicPitchPytorch(BaseTransform):

    def __init__(self, sr, device="cpu") -> None:
        super().__init__(sr, "basic_pitch")

        self.pt_model = BasicPitchTorch()

        file_path = pathlib.Path(__file__).parent.resolve()

        self.pt_model.load_state_dict(
            torch.load(
                os.path.join(
                    file_path,
                    'basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth'
                )))
        self.pt_model.eval()
        self.pt_model.to(device)
        self.device = device

    @torch.no_grad
    def __call__(self, waveform, **kwargs):
        if type(waveform) != torch.Tensor:
            waveform = torch.from_numpy(waveform).to(self.device)

        if self.sr != 22050:
            waveform = torchaudio.functional.resample(waveform=waveform,
                                                      orig_freq=self.sr,
                                                      new_freq=22050)

        #print(waveform)
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            results = []
            for wave in waveform:
                _, midi_data, _ = predict(model=self.pt_model,
                                          audio=wave.squeeze().cpu(),
                                          device=self.device)
                results.append(midi_data)
            return results
        else:
            _, midi_data, _ = predict(model=self.pt_model,
                                      audio=waveform.squeeze().cpu(),
                                      device=self.device)
            return midi_data


from scipy.signal import lfilter
from random import random


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)




class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


import pedalboard


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class PSTS(BaseTransform):

    def __init__(self,
                 sr,
                 ts_min=0.51,
                 ts_max=1.99,
                 pitch_min=-4,
                 pitch_max=+4,
                 chunk_size=None):
        super().__init__(sr, "pstc")
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.chunk_size = chunk_size

    def process_audio(self, audio):
        if self.pitch_min == self.pitch_max:
            pitch_shifts = 0
        else:
            if self.chunk_size is None:
                pitch_shifts = np.random.randint(self.pitch_min,
                                                 self.pitch_max, 1)[0]
            else:
                pitch_shifts = np.random.randint(
                    self.pitch_min, self.pitch_max,
                    audio.shape[-1] // self.chunk_size + 1)
                pitch_shifts = np.repeat(pitch_shifts, self.chunk_size)
                pitch_shifts = pitch_shifts[:audio.shape[-1]]

        if self.ts_min == self.ts_max:
            time_stretchs = 1.
        else:
            if self.chunk_size is None:
                time_stretchs = np.random.uniform(self.ts_min,
                                                  (self.ts_max - 1) / 2 + 1,
                                                  1)[0]
                if time_stretchs > 1.:
                    time_stretchs = 2 * (time_stretchs - 1) + 1
            else:
                time_stretchs = np.random.uniform(
                    self.ts_min, (self.ts_max - 1) / 2 + 1,
                    audio.shape[-1] // self.chunk_size + 1)

                time_stretchs[time_stretchs > 1.] = 2 * (
                    time_stretchs[time_stretchs > 1.] - 1) + 1

                time_stretchs = np.repeat(time_stretchs, self.chunk_size)
                time_stretchs = time_stretchs[:audio.shape[-1]]

        audio_transformed = pedalboard.time_stretch(
            audio,
            samplerate=self.sr,
            stretch_factor=time_stretchs,
            pitch_shift_in_semitones=pitch_shifts,
            use_time_domain_smoothing=True)
        return audio_transformed

    def __call__(self, audio):
        return self.process_audio(audio)

