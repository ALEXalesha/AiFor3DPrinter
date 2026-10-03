"""Сеть: MobileNetV3-Large (torchvision, BSD-3, предобучена на ImageNet) + голова-свёртка 1x1.

Голова без скрытого слоя: свёртка 1x1 даёт карту [5, h, w] (вклад каждой клетки в логит
каждого вида), логиты - её среднее. Тогда карта = Grad-CAM для этой головы, и её можно
отдать из ONNX вторым выходом: службе не нужен PyTorch для тепловой карты.
"""
import torch
import torch.nn as nn

from watcher.classes import CLASSES

INPUT_SIZE = (288, 160)     # ширина, высота


class AifpNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES), pretrained=True):
        super().__init__()
        import torchvision
        w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.mobilenet_v3_large(weights=w)
        self.features = m.features
        self.drop = nn.Dropout(0.25)
        self.head = nn.Conv2d(960, n_classes, 1)

    def forward(self, x):
        f = self.features(x)
        cam = self.head(self.drop(f))
        return cam.mean(dim=(2, 3)), cam


class Exportable(nn.Module):
    """Для ONNX: без dropout, два выхода logits и cam."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        f = self.net.features(x)
        cam = self.net.head(f)
        return cam.mean(dim=(2, 3)), cam


def param_groups(net, lr_backbone, lr_head):
    return [{"params": net.features.parameters(), "lr": lr_backbone},
            {"params": net.head.parameters(), "lr": lr_head}]


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    net = AifpNet(pretrained=False)
    net.load_state_dict(ck["model"])
    return net, ck
