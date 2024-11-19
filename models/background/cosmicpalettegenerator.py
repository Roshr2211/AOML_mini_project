import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

@threestudio.register("cosmic-palette-generator")
class CosmicPaletteGenerator(BaseBackground):
    @dataclass
    class Settings(BaseBackground.Config):
        output_channels: int = 3
        color_transform: str = "sigmoid"
        input_projection: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "neurons": 16,
                "layers": 2,
            }
        )
        use_random_variation: bool = False
        variation_probability: float = 0.5
        fixed_color: Optional[Tuple[float, float, float]] = None

    settings: Settings

    def initialize(self) -> None:
        self.projection = get_encoding(3, self.settings.input_projection)
        self.model = get_mlp(
            self.projection.n_output_dims,
            self.settings.output_channels,
            self.settings.network_config,
        )

    def generate(self, input_data: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        if not self.training and self.settings.fixed_color is not None:
            return torch.ones(*input_data.shape[:-1], self.settings.output_channels).to(
                input_data
            ) * torch.as_tensor(self.settings.fixed_color).to(input_data)

        # Normalize input_data before processing
        input_data = (input_data + 1.0) / 2.0  # (-1, 1) => (0, 1)
        embedded_input = self.projection(input_data.view(-1, 3))
        output = self.model(embedded_input).view(*input_data.shape[:-1], self.settings.output_channels)
        output = get_activation(self.settings.color_transform)(output)

        if (
            self.training
            and self.settings.use_random_variation
            and random.random() < self.settings.variation_probability
        ):
            # Apply random color variation with probability variation_probability
            output = output * 0 + (  # prevent checking for unused parameters in DDP
                torch.rand(input_data.shape[0], 1, 1, self.settings.output_channels)
                .to(input_data)
                .expand(*input_data.shape[:-1], -1)
            )

        if not self.training:
            output = torch.ones_like(output).to(input_data)

        return output
