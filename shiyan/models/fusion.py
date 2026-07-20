from __future__ import annotations

import torch
import torch.nn as nn


class ModalityReweighting(nn.Module):
    def __init__(self, text_dim: int, num_modalities: int = 3) -> None:
        super().__init__()
        self.weight_predictor = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, num_modalities),
        )

    def forward(
        self,
        text_feature: torch.Tensor,
        modality_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_predictor(text_feature), dim=1)
        fused = 0.0
        for index, feature in enumerate(modality_features):
            fused = fused + weights[:, index : index + 1] * feature
        return fused, weights


class GateModulation(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_feature: torch.Tensor, image_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(text_feature)
        return gate * image_feature, gate


class FiLMModulation(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Linear(text_dim, feature_dim)

    def forward(
        self,
        text_feature: torch.Tensor,
        image_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        gamma = self.gamma(text_feature)
        beta = self.beta(text_feature)
        modulated = gamma * image_feature + beta
        return modulated, (gamma, beta)
