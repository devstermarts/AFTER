import nn_tilde
import torch
import cached_conv as cc
from after.autoencoder import AutoEncoder

import gin
from absl import app, flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer("step", 1000000, "Step to load the model from")
flags.DEFINE_string("model_path", None, "Path of the trained model")

class AE(nn_tilde.Module):

    def __init__(self, model_name, step) -> None:
        super().__init__()
        config = os.path.join(model_name, "config.gin")

        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [config],
                [],
            )

        sr = gin.query_parameter("%SR")
        print(sr)

        model = AutoEncoder()
        d = torch.load(os.path.join(FLAGS.model_path, "checkpoint" + str(step) + ".pt"))
        model.load_state_dict(d["model_state"])

        self.model = model

        test_array = torch.zeros((3, 1, 4096))
        z, _ = self.model.encode(test_array)

        self.comp_ratio = test_array.shape[-1] // z.shape[-1]

        self.latent_size = gin.query_parameter("%LATENT_SIZE")
        self.target_channels = 1

        self.register_method(
            "encode",
            in_channels=self.target_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=self.comp_ratio,
            input_labels=['(signal) input 1'],
            output_labels=[f"latent {i}" for i in range(self.latent_size)],
            test_buffer_size=self.comp_ratio,
        )

        self.register_method("decode",
                             in_channels=self.latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=self.target_channels,
                             out_ratio=1,
                             input_labels=[
                                 f'(signal) Latent dimension {i+1}'
                                 for i in range(self.latent_size)
                             ],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

        self.register_method("forward",
                             in_channels=1,
                             in_ratio=1,
                             out_channels=1,
                             out_ratio=1,
                             input_labels=['(signal) input 1'],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)

        z = self.model.encoder(x)

        z, _ = self.model.bottleneck(z)

        x = self.model.decoder(z)

        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        return x

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)
        z = self.model.encoder(x)
        z, _ = self.model.bottleneck(z)

        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model.decoder(z)
        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)
        return x

def main(argv):
    cc.use_cached_conv(False)
    ae = AE(model_name=FLAGS.model_path, step=FLAGS.step)
    test_array = torch.zeros((3, 1, ae.comp_ratio * 8))
    z = ae.encode(test_array)
    x = ae.decode(z)

    ae.export_to_ts( os.path.join(FLAGS.model_path, "export.ts"))
    
    cc.use_cached_conv(True)
    ae = AE(model_name=FLAGS.model_path, step=FLAGS.step)
    
    test_array = torch.zeros((3, 1, ae.comp_ratio * 8))
    z = ae.encode(test_array)
    x = ae.decode(z)

    ae.export_to_ts(os.path.join(FLAGS.model_path, "export_stream.ts"))
    print("Success !")

if __name__ == "__main__":
    app.run(main)
