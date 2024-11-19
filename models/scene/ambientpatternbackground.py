from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

@threestudio.register("ambient-pattern-background")
class AmbientPatternBackground(BaseBackground):
    @dataclass
    class Settings(BaseBackground.Config):
        output_channels: int = 3
        texture_height: int = 64
        texture_width: int = 64
        color_function: str = "sigmoid"

    settings: Settings

    def initialize(self) -> None:
        self.pattern = nn.Parameter(
            torch.randn((1, self.settings.output_channels, self.settings.texture_height, self.settings.texture_width))
        )

    def convert_to_uv(self, directions: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 2"]:
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        xy = (x**2 + y**2) ** 0.5
        u = torch.atan2(xy, z) / torch.pi
        v = torch.atan2(y, x) / (torch.pi * 2) + 0.5
        uv = torch.stack([u, v], -1)
        return uv

    def render(self, directions: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        dirs_shape = directions.shape[:-1]
        uv_coordinates = self.convert_to_uv(directions.reshape(-1, directions.shape[-1]))
        uv_coordinates = 2 * uv_coordinates - 1  # rescale to [-1, 1] for grid_sample
        uv_coordinates = uv_coordinates.reshape(1, -1, 1, 2)
        color = (
            F.grid_sample(
                self.pattern,
                uv_coordinates,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
            .reshape(self.settings.output_channels, -1)
            .T.reshape(*dirs_shape, self.settings.output_channels)
        )
        color = get_activation(self.settings.color_function)(color)
        return color
