import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *

@threestudio.register("abstract-color-scene")
class AbstractColorScene(BaseBackground):
    @dataclass
    class Settings(BaseBackground.Config):
        output_channels: int = 3
        base_color: Tuple = (1.0, 1.0, 1.0)
        dynamic_color: bool = False
        use_random_variation: bool = False
        variation_probability: float = 0.5

    settings: Settings

    def setup(self) -> None:
        self.background_color: Float[Tensor, "Nc"]
        if self.settings.dynamic_color:
            self.background_color = nn.Parameter(
                torch.as_tensor(self.settings.base_color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "background_color", torch.as_tensor(self.settings.base_color, dtype=torch.float32)
            )

    def render(self, view_angles: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        color = (
            torch.ones(*view_angles.shape[:-1], self.settings.output_channels).to(view_angles)
            * self.background_color
        )
        if (
            self.training
            and self.settings.use_random_variation
            and random.random() < self.settings.variation_probability
        ):
            # Apply random color variation with defined probability
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                torch.rand(view_angles.shape[0], 1, 1, self.settings.output_channels)
                .to(view_angles)
                .expand(*view_angles.shape[:-1], -1)
            )
        return color
