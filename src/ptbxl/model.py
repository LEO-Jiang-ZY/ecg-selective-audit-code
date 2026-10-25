"""轻量 1D ResNet。"""
from __future__ import annotations

import torch
from torch import nn


class Residual1D(nn.Module):
    def __init__(self, channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=7, padding=3, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.skip = nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False) if stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return torch.relu(x + residual)


class ECGResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, base_channels: int = 32, blocks=(1, 1, 1), dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels), nn.ReLU(),
        )
        stages = []
        for stage, count in enumerate(blocks):
            stride = 2 if stage else 1
            stages.extend([Residual1D(base_channels, stride if i == 0 else 1, dropout) for i in range(count)])
        self.encoder = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(self.pool(x).squeeze(-1))


class ConvBlock1D(nn.Module):
    """U-Net 用的残差卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return torch.relu(x + residual)


class ECGResUNet1D(nn.Module):
    """1D residual U-Net，编码-解码后全局池化做多标签分类。"""

    def __init__(self, in_channels: int = 12, num_classes: int = 5, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock1D(in_channels, c1, dropout)
        self.enc2 = ConvBlock1D(c1, c2, dropout)
        self.enc3 = ConvBlock1D(c2, c3, dropout)
        self.bottleneck = ConvBlock1D(c3, c4, dropout)
        self.pool = nn.MaxPool1d(2)
        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(c3 * 2, c3, dropout)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(c2 * 2, c2, dropout)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(c1 * 2, c1, dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(self.global_pool(d1).squeeze(-1))


class Bottleneck1D(nn.Module):
    """ResNet50 的 1D bottleneck block。"""

    expansion = 4

    def __init__(self, in_channels: int, width: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        out_channels = width * self.expansion
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = torch.relu(self.norm1(self.conv1(x)))
        out = torch.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return torch.relu(self.dropout(out) + residual)


class ECGResNet50_1D(nn.Module):
    """标准 ResNet50 的 1D 版本（[3,4,6,3] bottleneck）。"""

    def __init__(self, in_channels: int = 12, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 3, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(512, 256, 6, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(2048, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, width: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [Bottleneck1D(in_channels, width, stride, dropout)]
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(width * 4, width, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).squeeze(-1))


class ECGTransformer1D(nn.Module):
    """Patch embedding + TransformerEncoder 的 1D ECG 分类器。"""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        patch_size: int = 25,
        stride: int = 25,
        input_length: int = 1000,
    ):
        super().__init__()
        self.patch = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=stride)
        num_patches = (input_length - patch_size) // stride + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.position[:, : x.size(1) + 1]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    """按配置中的 arch 字段构造模型；缺省为原轻量 1D ResNet。"""
    cfg = dict(model_cfg)
    arch = str(cfg.pop("arch", "resnet1d")).lower()
    if arch in {"resnet1d", "ecg_resnet1d"}:
        return ECGResNet1D(num_classes=num_classes, **cfg)
    if arch in {"resunet1d", "resunet"}:
        return ECGResUNet1D(num_classes=num_classes, **cfg)
    if arch in {"resnet50_1d", "resnet50"}:
        return ECGResNet50_1D(num_classes=num_classes, **cfg)
    if arch in {"transformer1d", "transformer"}:
        return ECGTransformer1D(num_classes=num_classes, **cfg)
    raise ValueError(f"未知 arch={arch!r}")
