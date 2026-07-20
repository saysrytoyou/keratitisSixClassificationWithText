from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import ImageEncoder, TextPriorEncoder
from .fusion import FiLMModulation, GateModulation, ModalityReweighting


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.dli_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.fsi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.sbi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim * 3, num_classes)

    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        dli, fsi, sbi = torch.chunk(image, chunks=3, dim=1)
        fused = torch.cat(
            [self.dli_encoder(dli), self.fsi_encoder(fsi), self.sbi_encoder(sbi)],
            dim=1,
        )
        return self.classifier(fused)


class TextOnlyModel(nn.Module):
    def __init__(self, text_dim: int, num_classes: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.text_encoder = TextPriorEncoder(input_dim=text_dim, hidden_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, image: torch.Tensor | None = None, text: torch.Tensor | None = None) -> torch.Tensor:
        if text is None:
            raise ValueError("text-only model requires text input")
        return self.classifier(self.text_encoder(text))


class LateFusionModel(nn.Module):
    def __init__(self, text_dim: int, num_classes: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.dli_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.fsi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.sbi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.text_encoder = TextPriorEncoder(input_dim=text_dim, hidden_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        if text is None:
            raise ValueError("late fusion model requires text input")
        dli, fsi, sbi = torch.chunk(image, chunks=3, dim=1)
        fused = torch.cat(
            [
                self.dli_encoder(dli),
                self.fsi_encoder(fsi),
                self.sbi_encoder(sbi),
                self.text_encoder(text),
            ],
            dim=1,
        )
        return self.classifier(fused)


class PriorGuidedModel(nn.Module):
    def __init__(self, text_dim: int, num_classes: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.dli_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.fsi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.sbi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.text_encoder = TextPriorEncoder(input_dim=text_dim, hidden_dim=feature_dim)
        self.modulation = GateModulation(text_dim=feature_dim, feature_dim=feature_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        if text is None:
            raise ValueError("prior-guided model requires text input")
        dli, fsi, sbi = torch.chunk(image, chunks=3, dim=1)
        dli_feat = self.dli_encoder(dli)
        fsi_feat = self.fsi_encoder(fsi)
        sbi_feat = self.sbi_encoder(sbi)
        image_feat = torch.cat([dli_feat, fsi_feat, sbi_feat], dim=1)
        text_feat = self.text_encoder(text)
        guided_image, _ = self.modulation(text_feat, image_feat)
        fused = torch.cat([guided_image, text_feat], dim=1)
        return self.classifier(fused)


class CPGMFNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_classes: int,
        feature_dim: int = 256,
        modulation_type: str = "film",
    ) -> None:
        super().__init__()
        self.dli_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.fsi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.sbi_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.text_encoder = TextPriorEncoder(input_dim=text_dim, hidden_dim=feature_dim)
        self.reweighting = ModalityReweighting(text_dim=feature_dim, num_modalities=3)
        if modulation_type == "gate":
            self.modulation = GateModulation(text_dim=feature_dim, feature_dim=feature_dim)
        elif modulation_type == "film":
            self.modulation = FiLMModulation(text_dim=feature_dim, feature_dim=feature_dim)
        else:
            raise ValueError(f"unsupported modulation_type: {modulation_type}")
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        if text is None:
            raise ValueError("CPGMFNet requires text input")
        dli, fsi, sbi = torch.chunk(image, chunks=3, dim=1)
        dli_feat = self.dli_encoder(dli)
        fsi_feat = self.fsi_encoder(fsi)
        sbi_feat = self.sbi_encoder(sbi)
        text_feat = self.text_encoder(text)
        image_feat, _ = self.reweighting(text_feat, [dli_feat, fsi_feat, sbi_feat])
        modulated, _ = self.modulation(text_feat, image_feat)
        fused = torch.cat([modulated, text_feat], dim=1)
        return self.classifier(fused)


def build_model(model_name: str, text_dim: int, num_classes: int, feature_dim: int = 256) -> nn.Module:
    if model_name == "image":
        return ImageOnlyModel(num_classes=num_classes, feature_dim=feature_dim)
    if model_name == "text":
        return TextOnlyModel(text_dim=text_dim, num_classes=num_classes, feature_dim=feature_dim)
    if model_name == "late":
        return LateFusionModel(text_dim=text_dim, num_classes=num_classes, feature_dim=feature_dim)
    if model_name == "prior":
        return PriorGuidedModel(text_dim=text_dim, num_classes=num_classes, feature_dim=feature_dim)
    if model_name == "cpgmf_gate":
        return CPGMFNet(text_dim=text_dim, num_classes=num_classes, feature_dim=feature_dim, modulation_type="gate")
    if model_name == "cpgmf_film":
        return CPGMFNet(text_dim=text_dim, num_classes=num_classes, feature_dim=feature_dim, modulation_type="film")
    raise ValueError(f"unsupported model: {model_name}")
