from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops_exts import rearrange_many
from torch.optim import AdamW, Adam
from .core import DistanceWrap
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm
import gin
import os


class Dummy():

    def __getattr__(self, key):

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


@gin.configurable
class Trainer(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 waveform_losses: List[Tuple[int, nn.Module]] = [],
                 reg_losses: List[Tuple[int, nn.Module]] = [],
                 multiband_distances: List[Tuple[int, nn.Module]] = [],
                 sr: int = 16000,
                 max_steps: int = 1000000,
                 discriminator=None,
                 warmup_steps=0,
                 freeze_encoder_step=1000000000,
                 device="cpu",
                 update_discriminator_every: int = 3):

        super().__init__()

        self.waveform_losses = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in waveform_losses
        ]).to(device)
        self.reg_losses = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in reg_losses
        ]).to(device)
        self.multiband_distances = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in multiband_distances
        ]).to(device) if len(multiband_distances) > 0 else []

        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.sr = sr
        self.max_steps = max_steps
        self.warmup = False
        self.warmup_steps = warmup_steps
        self.freeze_encoder_step = freeze_encoder_step
        self.step = 0
        self.device = device
        self.update_discriminator_every = update_discriminator_every

        self.init_opt()

    def compute_loss(self, x, y, z=None, x_multiband=None, y_multiband=None):
        total_loss = 0.

        losses = {}
        for dist in self.waveform_losses:
            loss_value = dist(x, y)
            losses[dist.name] = loss_value.item()
            total_loss += loss_value * dist.scale

        if z is not None:
            for dist in self.reg_losses:
                loss_value = dist(z)
                losses[dist.name] = loss_value.item()
                total_loss += loss_value * dist.scale

        if x_multiband is not None and y_multiband is not None:
            for dist in self.multiband_distances:
                loss_value = dist(x_multiband, y_multiband)
                losses[dist.name + "_multiband"] = loss_value.item()
                total_loss += loss_value * dist.scale

        losses["total_loss"] = torch.tensor(total_loss).item()

        return total_loss, losses

    def get_losses_names(self):
        names = []
        for loss in self.reg_losses + self.waveform_losses:
            names.append(loss.name)
        names.extend(["total_loss"])

        if True:  #self.model.pqmf_bands > 1:
            for loss in self.multiband_distances:
                names.append(loss.name + "_multiband")

        if self.discriminator is not None:
            names.extend(self.discriminator.get_losses_names())
        return names

    def init_opt(self, lr=1e-4):
        print("warning, putting all models paramters")
        self.opt = AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt,
                                                                gamma=0.999996)

        if self.discriminator is not None:
            self.opt_dis = AdamW(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(0.8, 0.9))
            self.scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(
                self.opt, gamma=0.999996)

    def load_model(self, path, step, load_discrim=False):
        d = torch.load(path + "/checkpoint" + str(step) + ".pt")
        self.model.load_state_dict(d["model_state"])
        self.opt.load_state_dict(d["opt_state"])

        if load_discrim == True:
            self.discriminator.load_state_dict(d["dis_state"])
            self.opt_dis.load_state_dict(d["opt_dis_state"])
        self.step = step + 1

    def training_step(self, x):

        self.train()

        if True:
            if (self.discriminator is not None and self.warmup
                ) and self.step % self.update_discriminator_every == 0:

                with torch.no_grad():
                    z, x_multiband = self.model.encode(x, with_multi=True)
                    y, y_multiband = self.model.decode(z, with_multi=True)
                    """loss_ae, loss_out = self.compute_loss(x,
                                                        y,
                                                        z=z,
                                                        x_multiband=x_multiband,
                                                        y_multiband=y_multiband)"""
                loss_out = {}
                loss_gen, loss_dis, loss_dis_dict = self.discriminator.compute_losses(
                    x, y)

                self.opt_dis.zero_grad()
                loss_dis.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                               2.0)
                self.opt_dis.step()
                loss_out.update(loss_dis_dict)

            else:
                if self.step > self.freeze_encoder_step:
                    with torch.no_grad():
                        z, x_multiband = self.model.encode(x, with_multi=True)
                        z = z.detach()
                else:
                    z, x_multiband = self.model.encode(x, with_multi=True)

                y, y_multiband = self.model.decode(z, with_multi=True)
                loss_ae, loss_out = self.compute_loss(x,
                                                      y,
                                                      z=z,
                                                      x_multiband=x_multiband,
                                                      y_multiband=y_multiband)

                if self.warmup:
                    loss_gen, loss_dis, loss_dis_dict = self.discriminator.compute_losses(
                        x, y)
                else:
                    loss_gen = 0.
                    loss_dis_dict = {
                        k: 0.
                        for k in self.discriminator.get_losses_names()
                    }

                loss_gen = loss_gen + loss_ae

                loss_out.update(loss_dis_dict)
                self.opt.zero_grad()
                loss_gen.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.opt.step()

        return loss_out

    def val_step(self, validloader, get_audio=False, get_losses=True):

        tval = tqdm(range(len(validloader)), unit="batch")

        self.eval()
        all_losses = {key: 0 for key in self.get_losses_names()}
        with torch.no_grad():
            for i, x in enumerate(validloader):
                x = x.to(self.device)
                z, x_multiband = self.model.encode(x, with_multi=True)
                y, y_multiband = self.model.decode(z, with_multi=True)

                _, losses = self.compute_loss(x, y, z=z)

                for k, v in losses.items():
                    all_losses[k] += v

                tval.update(1)

                if get_losses == False:
                    break

                if i == 50:
                    break

            all_losses = {
                k: v / len(validloader)
                for k, v in all_losses.items()
            }
            if get_audio:
                x, y = x.cpu()[:4], y.cpu()[:4]
                audio = torch.cat(
                    (x, torch.zeros(
                        (x.shape[0], x.shape[1], int(self.sr / 3))), y),
                    dim=-1)
                audio = audio.reshape(1, 1, -1)
                if get_losses == False:
                    return audio
                return all_losses, audio
            else:
                return all_losses, None

    def fit(self,
            trainloader,
            validloader,
            tensorboard=None,
            steps_display=20):

        if tensorboard is not None:
            logger = SummaryWriter(log_dir=tensorboard)
        else:
            logger = Dummy()

        tepoch = tqdm(range(self.max_steps), unit="batch")
        all_losses_sum = {key: 0 for key in self.get_losses_names()}
        all_losses_count = {key: 0 for key in self.get_losses_names()}

        with open(os.path.join(tensorboard, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        while self.step < self.max_steps:
            for x in trainloader:
                x = x.to(self.device)

                all_losses = self.training_step(x)

                for k in all_losses:
                    all_losses_sum[k] += all_losses[k]
                    all_losses_count[k] += 1

                tepoch.update(1)

                if not self.step % steps_display:
                    tepoch.set_postfix(loss=all_losses_sum["total_loss"] /
                                       steps_display)
                    for k in all_losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          all_losses_sum[k] /
                                          all_losses_count[k],
                                          global_step=self.step)
                        all_losses_sum[k] = 0.
                        all_losses_count[k] = 0

                if (not self.step % 10000):
                    print("Validation Step")

                    if validloader is not None:
                        all_losses, audio = self.val_step(validloader,
                                                          get_audio=True)
                        print("Validation Loss at step ", self.step, " : ",
                              all_losses["total_loss"])
                        if logger:
                            for k, v in all_losses.items():
                                logger.add_scalar('Validation/' + k,
                                                  v,
                                                  global_step=self.step)

                            logger.add_audio("Validation/Audio",
                                             audio,
                                             global_step=self.step,
                                             sample_rate=self.sr)
                    else:
                        audio = self.val_step(trainloader,
                                              get_audio=True,
                                              get_losses=False)
                        logger.add_audio("Validation/Audio",
                                         audio,
                                         global_step=self.step,
                                         sample_rate=self.sr)

                    d = {
                        "model_state": self.model.state_dict(),
                        "opt_state": self.opt.state_dict(),
                        "dis_state": self.discriminator.state_dict(),
                        "opt_dis_state": self.opt_dis.state_dict()
                    }

                    if not (self.step % 50000):
                        torch.save(
                            d, tensorboard + "/checkpoint" + str(self.step) +
                            ".pt")

                if self.step > self.max_steps + 1000:
                    exit()

                if self.step > self.warmup_steps and self.warmup == False:
                    self.warmup = True
                    print("Warmup finished")

                self.step += 1
