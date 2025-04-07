import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord
from ..core import mean_difference
import gin

import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange
from scipy import signal

from torch.nn.utils import weight_norm


def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen


def feature_matching_function(x, y, normalize_losses):
    distance = abs(x - y).mean()

    if normalize_losses:
        scale = abs(x).mean()
        return distance / scale
    return distance


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int],
                   dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 max_filters: int = 1024,
                 filters_scale: int = 1,
                 kernel_size: tp.Tuple[int, int] = (3, 9),
                 dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2),
                 normalized: bool = True,
                 activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2},
                 spec_scale_pow=0.0,
                 window="hann"):
        super().__init__()
        print(stride)
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels,
                       self.filters,
                       kernel_size=kernel_size,
                       padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(in_chs,
                           out_chs,
                           kernel_size=kernel_size,
                           stride=stride,
                           dilation=(dilation, 1),
                           padding=get_2d_padding(kernel_size, (dilation, 1))))
            in_chs = out_chs
        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(in_chs,
                       out_chs,
                       kernel_size=(kernel_size[0], kernel_size[0]),
                       padding=get_2d_padding(
                           (kernel_size[0], kernel_size[0]))))
        self.conv_post = NormConv2d(out_chs,
                                    self.out_channels,
                                    kernel_size=(kernel_size[0],
                                                 kernel_size[0]),
                                    padding=get_2d_padding(
                                        (kernel_size[0], kernel_size[0])))

        self.spec_scale_pow = spec_scale_pow

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        if self.spec_scale_pow != 0.0:
            z = z * torch.pow(z.abs() + 1e-6, self.spec_scale_pow)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512],
                 hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512],
                 spec_scale_pow: float = 0.0,
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              n_fft=n_ffts[i],
                              win_length=win_lengths[i],
                              hop_length=hop_lengths[i],
                              spec_scale_pow=spec_scale_pow,
                              **kwargs) for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


@gin.configurable
class EncodecDiscriminator(nn.Module):

    def __init__(self,
                 filters=64,
                 n_ffts=[2048, 1024, 512, 256, 128],
                 hop_lengths=[512, 256, 128, 64, 32],
                 win_lengths=[2048, 1024, 512, 256, 128],
                 weights={},
                 normalize_losses=True,
                 spec_scale_pow=0.0):

        super().__init__()

        self.discriminators = MultiScaleSTFTDiscriminator(
            filters=filters,
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            win_lengths=win_lengths,
            spec_scale_pow=spec_scale_pow)
        self.weights = weights
        self.normalize_losses = normalize_losses

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def get_losses_names(self):
        return [
            "feature_matching", "pred_real", "pred_fake", "discriminator",
            "adversarial"
        ]

    def compute_losses(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)

        pred_fake, pred_true = 0., 0.

        for i, (scale_true,
                scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: feature_matching_function(
                        x, y, self.normalize_losses),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = hinge_gan(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

            pred_fake += logits_fake[i].mean()
            pred_true += logits_true[i].mean()

        dis_loss = dis_loss / len(logits_true)
        adv_loss = adv_loss / len(logits_true)
        feature_matching_distance = feature_matching_distance / len(
            logits_true)

        loss_dict = {
            "discriminator": dis_loss.item(),
            "adversarial": adv_loss.item(),
            "feature_matching": feature_matching_distance.item(),
            "pred_real": pred_true.mean().item(),
            "pred_fake": pred_fake.mean().item()
        }
        loss_gen = adv_loss * self.weights[
            "adversarial"] + feature_matching_distance * self.weights[
                "feature_matching"]

        return loss_gen, dis_loss, loss_dict
