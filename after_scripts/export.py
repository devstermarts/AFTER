import torch.nn as nn
import nn_tilde
import torch

import argparse
import os
from after.diffusion import RectifiedFlow

torch.set_grad_enabled(False)

import gin
import cached_conv as cc
from absl import flags, app

cc.use_cached_conv(True)
parser = argparse.ArgumentParser()

FLAGS = flags.FLAGS
# Flags definition
flags.DEFINE_string("name",
                    default="./after_runs/test",
                    help="Name of the experiment folder")
flags.DEFINE_integer("step", default=0, help="Step number of checkpoint")
flags.DEFINE_string("out_name", default=None, help="Output name")
flags.DEFINE_string("emb_model_path_encode",
                    default="./pretrained/test.ts",
                    help="Path to encoder model")
flags.DEFINE_string("emb_model_path_decode",
                    default="./pretrained/test.ts",
                    help="Path to decoder model")
flags.DEFINE_integer("chunk_size", default=4, help="Chunk size")
flags.DEFINE_integer("max_cache_size", default=128, help="Max cache size")


def main(argv):
    # Parse model folder
    folder = FLAGS.name
    checkpoint_path = folder + "/checkpoint" + str(FLAGS.step) + "_EMA.pt"
    config = folder + "/config.gin"

    out_name = FLAGS.out_name if FLAGS.out_name is not None else FLAGS.name

    # Parse config
    gin.parse_config_file(config)
    SR = gin.query_parameter("%SR")

    with gin.unlock_config():
        with gin.unlock_config():
            gin.bind_parameter("transformer.Denoiser.max_cache_size",
                               FLAGS.max_cache_size)

    # Emb model

    # Instantiate model
    blender = RectifiedFlow()

    # Load checkpoints
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    blender.load_state_dict(state_dict, strict=False)

    # Emb model
    # Send to device
    blender = blender.eval()

    # Get some parameters
    n_signal = gin.query_parameter('%N_SIGNAL')
    n_signal_timbre = gin.query_parameter('%N_SIGNAL')
    zt_channels = gin.query_parameter("%ZT_CHANNELS")
    zs_channels = gin.query_parameter("%ZS_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = blender.net
            self.encoder = blender.encoder
            self.encoder_time = blender.encoder_time

            self.n_signal = n_signal
            self.n_signal_timbre = n_signal_timbre
            self.chunk_size = FLAGS.chunk_size
            self.zs_channels = zs_channels
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_out = torch.jit.load(
                FLAGS.emb_model_path_decode).eval()
            self.emb_model_structure = torch.jit.load(
                FLAGS.emb_model_path_encode).eval()
            self.emb_model_timbre = torch.jit.load(
                FLAGS.emb_model_path_encode).eval()

            self.drop_value = blender.drop_value

            # Get the ae ratio
            dummy = torch.zeros(1, 1, 4 * 4096)
            z = self.emb_model_out.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.sr = gin.query_parameter("%SR")
            self.zt_buffer = self.n_signal_timbre * self.ae_ratio

            ## ATTRIBUTES ##
            self.register_attribute("nb_steps", 6)
            self.register_attribute("guidance", 1.)
            self.register_attribute("guidance_timbre", 1.)
            self.register_attribute("guidance_structure", 1.)
            self.register_attribute("learn_zsem", False)

            ## BUFFERS ##
            self.register_buffer(
                "previous_timbre",
                torch.zeros(4, self.ae_latents, self.n_signal_timbre))

            self.register_buffer("last_zsem", torch.zeros(4, self.zt_channels))

            ## METHODS ##
            self.register_method(
                "forward",
                in_channels=2,
                in_ratio=1,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input structure",
                    f"(signal) Input timbre",
                ],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "structure",
                in_channels=1,
                in_ratio=1,
                out_channels=self.zs_channels,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input structure",
                ],
                output_labels=[
                    f"(signal) Output structure {i}"
                    for i in range(zs_channels)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "timbre",
                in_channels=1,
                in_ratio=1,
                out_channels=self.zt_channels,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input timbre",
                ],
                output_labels=[
                    f"(signal) Output timbre {i}" for i in range(zt_channels)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate",
                in_channels=zt_channels + zs_channels,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input structure {i}" for i in range(zs_channels)
                ] + [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "diffuse",
                in_channels=zt_channels + zs_channels,
                in_ratio=self.ae_ratio,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input structure {i}"
                    for i in range(self.zs_channels)
                ] + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "decode",
                in_channels=self.ae_latents,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Latent input {i}"
                    for i in range(self.ae_latents)
                ],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

        @torch.jit.export
        def get_learn_zsem(self) -> bool:
            return self.learn_zsem[0]

        @torch.jit.export
        def set_learn_zsem(self, learn_zsem: bool) -> int:
            self.learn_zsem = (learn_zsem, )
            return 0

        @torch.jit.export
        def get_guidance(self) -> float:
            return self.guidance[0]

        @torch.jit.export
        def set_guidance(self, guidance: float) -> int:
            self.guidance = (guidance, )
            return 0

        @torch.jit.export
        def get_guidance_timbre(self) -> float:
            return self.guidance_timbre[0]

        @torch.jit.export
        def set_guidance_timbre(self, guidance_timbre: float) -> int:
            self.guidance_timbre = (guidance_timbre, )
            return 0

        @torch.jit.export
        def get_guidance_structure(self) -> float:
            return self.guidance_structure[0]

        @torch.jit.export
        def set_guidance_structure(self, guidance_structure: float) -> int:
            self.guidance_structure = (guidance_structure, )
            return 0

        @torch.jit.export
        def get_nb_steps(self) -> int:
            return self.nb_steps[0]

        @torch.jit.export
        def set_nb_steps(self, nb_steps: int) -> int:
            self.nb_steps = (nb_steps, )
            return 0

        @torch.jit.ignore
        def model_forward_dummy(self, x: torch.Tensor, time: torch.Tensor,
                                cond: torch.Tensor, time_cond: torch.Tensor,
                                cache_index: int) -> torch.Tensor:

            guidance_timbre = self.guidance_timbre[0]
            guidance_structure = self.guidance_structure[0]
            print(guidance_timbre, guidance_structure)

            full_time = time.repeat(4, 1, 1)
            full_x = x.repeat(4, 1, 1)

            full_cond = torch.cat([
                cond, self.drop_value * torch.ones_like(cond), cond,
                self.drop_value * torch.ones_like(cond)
            ])

            full_time_cond = torch.cat([
                time_cond,
                time_cond,
                self.drop_value * torch.ones_like(time_cond),
                self.drop_value * torch.ones_like(time_cond),
            ])

            dx = self.net(full_x,
                          time=full_time,
                          cond=full_cond,
                          time_cond=full_time_cond,
                          cache_index=cache_index)

            dx_full, dx_time_cond, dx_cond, dx_none = torch.chunk(dx, 4, dim=0)

            total_guidance = 0.5 * (guidance_structure + guidance_timbre)

            guidance_cond_factor = guidance_timbre / (guidance_structure +
                                                      guidance_timbre)

            guidance_joint_factor = 1 - 2 * (abs(guidance_cond_factor - 0.5))

            print(total_guidance, guidance_cond_factor, guidance_joint_factor)

            dx = dx_none + total_guidance * (guidance_joint_factor *
                                             (dx_full - dx_none) +
                                             (1 - guidance_joint_factor) *
                                             (guidance_cond_factor *
                                              (dx_cond - dx_none) +
                                              (1 - guidance_cond_factor) *
                                              (dx_time_cond - dx_none)))

            return dx

        def model_forward(self, x: torch.Tensor, time: torch.Tensor,
                          cond: torch.Tensor, time_cond: torch.Tensor,
                          cache_index: int) -> torch.Tensor:

            guidance_timbre = self.guidance_timbre[0]
            guidance_structure = self.guidance_structure[0]
            print(guidance_timbre, guidance_structure)

            full_time = time.repeat(2, 1, 1)
            full_x = x.repeat(2, 1, 1)

            full_cond = torch.cat(
                [cond, self.drop_value * torch.ones_like(cond)])

            full_time_cond = torch.cat([
                time_cond,
                time_cond,
            ])

            dx = self.net(full_x,
                          time=full_time,
                          cond=full_cond,
                          time_cond=full_time_cond,
                          cache_index=cache_index)

            dx_full, dx_none = torch.chunk(dx, 2, dim=0)
            dx = dx_none + (dx_full - dx_none) * guidance_timbre

            return dx

        def sample(self, x_last: torch.Tensor, cond: torch.Tensor,
                   time_cond: torch.Tensor):

            x = x_last
            t = torch.linspace(0, 1, self.nb_steps[0] + 1)
            dt = 1 / self.nb_steps[0]
            print("x is shape", x.shape)
            for i, t_value in enumerate(t[:-1]):

                dt = dt

                x = x + self.model_forward(x=x,
                                           time=t_value.repeat(
                                               x.shape[0], 1, x.shape[-1]),
                                           cond=cond,
                                           time_cond=time_cond,
                                           cache_index=i) * dt

                self.net.roll_cache(x.shape[-1], i)
            return x

        @torch.jit.export
        def timbre(self, x) -> torch.Tensor:
            x = self.emb_model_timbre.encode(x)
            self.previous_timbre[:x.shape[0]] = torch.cat(
                (self.previous_timbre[:x.shape[0]], x), -1)[..., x.shape[-1]:]

            zsem = self.encoder.forward_stream(
                self.previous_timbre[:x.shape[0]])
            self.last_zsem = zsem
            return zsem.unsqueeze(-1).repeat((1, 1, self.chunk_size))

        @torch.jit.export
        def structure(self, x) -> torch.Tensor:
            x = self.emb_model_structure.encode(x)
            x = self.encoder_time.forward_stream(x)
            return x

        @torch.jit.export
        def diffuse(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, -self.zt_channels:].mean(-1)
            time_cond = x[:, :self.zs_channels]
            x = torch.randn(n, self.ae_latents, x.shape[-1])
            x = self.sample(x[:1], time_cond=time_cond[:1], cond=zsem[:1])

            if n > 1:
                x = x.repeat(n, 1, 1)
            return x

        @torch.jit.export
        def decode(self, x: torch.Tensor) -> torch.Tensor:
            audio = self.emb_model_out.decode(x)
            return audio

        @torch.jit.export
        def generate(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse(x)
            audio = self.decode(z)
            return audio

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_structure, x_timbre = x[:, :1], x[:, 1:]
            structure, timbre = self.structure(x_structure), self.timbre(
                x_timbre)
            x = torch.cat((structure, timbre), 1)
            z = self.diffuse(x)
            audio = self.decode(z)
            return audio

    ####

    streamer = Streamer()

    dummmy = torch.randn(1, zs_channels + zt_channels, 128)
    out = streamer.diffuse(dummmy)
    os.makedirs("./exports/", exist_ok=True)
    streamer.export_to_ts("./exports/" + out_name + ".ts")

    print("Bravo - Export successful")


if __name__ == "__main__":
    app.run(main)
