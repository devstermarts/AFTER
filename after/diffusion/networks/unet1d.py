import torch
import torch.nn as nn
from .blocks import SelfAttention1d, CrossAttention1d
from einops import rearrange, reduce, repeat
import gin

class SPE(torch.nn.Module):

    def __init__(self, dim=128, max_positions=10000, scale=32):
        super().__init__()
        self.embedding_size = dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, x):
        x = x.view(-1)
        x = x * self.scale
        freqs = torch.arange(start=0,
                             end=self.embedding_size // 2,
                             dtype=torch.float32,
                             device=x.device)
        w = (1 / self.max_positions)**(2 * freqs / self.embedding_size)
        w = w[None, :]
        x = x[:, None]
        x = torch.cat([torch.sin(w * x), torch.cos(w * x)], dim=-1)
        return x.squeeze(1)



@gin.configurable
class ConvBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 skip_channels,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 act=nn.SiLU,
                 res=True):
        super().__init__()
        self.res = res
        self.cond_channels = cond_channels

        self.conv1 = nn.Conv1d(in_c + skip_channels + time_cond_channels,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")

        try:
            self.gn1 = nn.GroupNorm(
                min(16, (in_c + skip_channels + time_cond_channels) // 4),
                in_c + skip_channels + time_cond_channels)
        except:
            self.gn1 = nn.Identity()
            print("Warning, replacing group norm by identity")

        self.conv2 = nn.Conv1d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")

        try:
            self.gn2 = nn.GroupNorm(min(16, out_c // 4), out_c)
        except:
            self.gn2 = nn.Identity()
            print("Warning, replacing group norm by identity")

        self.act = act()

        self.time_mlp = nn.Sequential(nn.Linear(time_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))

        if self.cond_channels > 0:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_channels, 128), act(),
                                          nn.Linear(128, 2 * out_c))

        if skip_channels:
            self.to_out = nn.Conv1d(in_c, out_c, kernel_size=1, padding="same")
        else:
            self.to_out = nn.Identity()

    def forward(self, x, time=None, skip=None, cond=None, time_cond=None):
        res = x.clone()

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        if time_cond is not None:
            x = torch.cat([x, time_cond], dim=1)

        x = self.gn1(x)
        x = self.act(x)

        x = self.conv1(x)

        # Time modulation
        time = self.time_mlp(time)
        time_mult, time_add = torch.chunk(time, chunks=2, dim=1)
        x = x * time_mult[:, :, None] + time_add[:, :, None]

        # Condition modulation
        if self.cond_channels > 0:
            cond = self.cond_mlp(cond)
            cond_mult, cond_add = torch.chunk(cond, chunks=2, dim=1)
            x = x * cond_mult[:, :, None] + cond_add[:, :, None]

        x = self.gn2(x)
        x = self.act(x)

        x = self.conv2(x)

        if self.res:
            return x + self.to_out(res)

        return x


@gin.configurable
class EncoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size=3,
                 ratio=2,
                 act=nn.SiLU,
                 use_self_attn=False,
                 cross_attn_channels=0):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                skip_channels=0,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                act=act)

        self.self_attn = SelfAttention1d(
            in_c, 4) if use_self_attn else nn.Identity()
        self.cross_attn_channels = cross_attn_channels

        if ratio == 1:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  padding="same")
        else:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  stride=ratio,
                                  padding=(kernel_size) // 2)

    def forward(self, inputs, time, cond=None, time_cond=None):
        skip = self.conv(inputs, time=time, cond=cond, time_cond=time_cond)
        skip = self.self_attn(skip)
        x = self.pool(skip)
        return x, skip


@gin.configurable
class MiddleBlock1D(nn.Module):

    def __init__(
        self,
        in_c,
        time_cond_channels,
        time_channels,
        cond_channels,
        kernel_size=3,
        ratio=2,
        act=nn.SiLU,
        use_self_attn=False,
    ):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                skip_channels=0,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                act=act)
        self.self_attn = SelfAttention1d(
            in_c, in_c // 32) if use_self_attn else nn.Identity()

    def forward(self, x, time, time_cond=None, cond=None):
        x = self.conv(x, time=time, cond=cond, time_cond=time_cond)
        x = self.self_attn(x)
        return x


@gin.configurable
class DecoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 res=True,
                 act=nn.SiLU,
                 ratio=2,
                 use_self_attn=False,
                 skip_size=None):
        super().__init__()
        if ratio == 1:
            if in_c == out_c:
                self.up = nn.Identity()
            else:
                self.up = nn.Conv1d(in_c,
                                    out_c,
                                    kernel_size=3,
                                    stride=1,
                                    padding="same")
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=ratio),
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=1,
                          padding="same"))

        self.conv = ConvBlock1D(
            in_c=out_c,
            out_c=out_c,
            time_cond_channels=time_cond_channels,
            skip_channels=skip_size if skip_size is not None else out_c,
            time_channels=time_channels,
            cond_channels=cond_channels,
            kernel_size=kernel_size,
            res=res,
            act=act)

        self.self_attn = SelfAttention1d(
            out_c, 4) if use_self_attn else nn.Identity()

    def forward(self, x, skip, time, time_cond=None, cond=None, context=None):
        x = self.up(x)

        x = self.conv(x, time=time, skip=skip, cond=cond, time_cond=time_cond)

        x = self.self_attn(x)
        return x


@gin.configurable
class UNET1D(nn.Module):

    def __init__(self,
                 in_size=128,
                 out_size=None,
                 channels=[128, 128, 256, 256],
                 ratios=[2, 2, 2, 2, 2],
                 kernel_size=5,
                 time_channels=64,
                 time_cond_in_channels=1,
                 time_cond_channels=64,
                 cond_channels=32,
                 n_attn_layers=0,
                 use_res_last=False):

        super().__init__()

        self.channels = channels
        self.time_cond_channels = time_cond_channels
        self.time_cond_in_channels = time_cond_in_channels
        self.cond_channels = cond_channels
        self.time_channels = time_channels

        self.in_size = in_size
        if out_size is None:
            out_size = in_size

        n = len(self.channels)
        ratios = [1] + ratios

        if time_channels == 0:
            self.time_emb = lambda _: torch.empty(0)
        else:
            self.time_emb = SPE(time_channels)

        comp_ratios = []
        cur_ratio = 1
        for r in ratios:
            cur_ratio *= r
            comp_ratios.append(cur_ratio)

        if time_cond_channels:
            self.cond_emb_time = nn.ModuleList()
            self.cond_emb_time.append(
                nn.Sequential(
                    nn.Conv1d(time_cond_in_channels,
                              time_cond_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding="same"), nn.SiLU()))
            for i in range(0, n):
                self.cond_emb_time.append(
                    nn.Sequential(
                        nn.Conv1d(time_cond_channels,
                                  time_cond_channels,
                                  kernel_size=kernel_size,
                                  stride=ratios[i],
                                  padding="same" if ratios[i] == 1 else
                                  (kernel_size) // 2), nn.SiLU()))

        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()

        self.down_layers.append(
            EncoderBlock1D(
                in_c=in_size +
                (time_cond_in_channels if not time_cond_channels else 0),
                out_c=channels[0],
                time_channels=time_channels,
                time_cond_channels=time_cond_channels,
                cond_channels=cond_channels,
                kernel_size=kernel_size,
                ratio=ratios[0]))

        for i in range(1, n):
            self.down_layers.append(
                EncoderBlock1D(
                    in_c=channels[i - 1],
                    out_c=channels[i],
                    time_channels=time_channels,
                    time_cond_channels=time_cond_channels,
                    cond_channels=cond_channels,
                    kernel_size=kernel_size,
                    ratio=ratios[i],
                    use_self_attn=i >= n - n_attn_layers,
                ))
            self.up_layers.append(
                DecoderBlock1D(
                    in_c=channels[n - i],
                    out_c=channels[n - i - 1],
                    time_channels=time_channels,
                    time_cond_channels=time_cond_channels,
                    cond_channels=cond_channels,
                    kernel_size=kernel_size,
                    ratio=ratios[n - i],
                    use_self_attn=i <= (n_attn_layers),
                ))

        self.up_layers.append(
            DecoderBlock1D(
                in_c=channels[0],
                out_c=out_size,
                skip_size=in_size +
                (time_cond_in_channels if not time_cond_channels else 0),
                time_channels=time_channels,
                time_cond_channels=time_cond_channels,
                cond_channels=cond_channels,
                kernel_size=kernel_size,
                res=use_res_last,
                ratio=ratios[0]))

        self.middle_block = MiddleBlock1D(
            in_c=channels[-1],
            time_channels=time_channels,
            cond_channels=cond_channels,
            time_cond_channels=time_cond_channels,
            kernel_size=kernel_size,
            use_self_attn=n_attn_layers > 0)

    def forward(self, x, time=None, time_cond=None, cond=None, time_emb=None):

        if time_emb == None:
            time_emb = self.time_emb(time).to(x)

        skips = []
        time_conds = []

        if self.time_cond_channels:
            for layer, cond_layer in zip(self.down_layers, self.cond_emb_time):

                if time_cond is not None:
                    time_cond = cond_layer(time_cond)

                x, skip = layer(x,
                                time=time_emb,
                                time_cond=time_cond,
                                cond=cond)

                time_conds.append(time_cond)
                skips.append(skip)

            time_cond = self.cond_emb_time[-1](time_cond)

            x = self.middle_block(x,
                                  time=time_emb,
                                  time_cond=time_cond,
                                  cond=cond)

            for layer in self.up_layers:
                skip = skips.pop(-1)
                time_cond = time_conds.pop(-1)

                x = layer(x,
                          skip=skip,
                          time=time_emb,
                          time_cond=time_cond,
                          cond=cond)

            return x

        else:
            if self.time_cond_in_channels:
                x = torch.cat([x, time_cond], dim=1)
            for layer in self.down_layers:
                x, skip = layer(x, time=time_emb, cond=cond)
                skips.append(skip)

            x = self.middle_block(x, time=time_emb, cond=cond)

            for layer in self.up_layers:

                skip = skips.pop(-1)

                x = layer(x, skip, time=time_emb, cond=cond)
            return x
