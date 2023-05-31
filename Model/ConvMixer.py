import torch.nn as nn
import torchsummary

from Model.LayerModules import DepthWiseConv, PointWiseConv


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=2):
        super(ConvMixer, self).__init__()
        self.depth = depth
        self.dim = dim

        self.conv = nn.Conv2d(3, dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

        MixerLayer = nn.Sequential(
            DepthWiseConv(dim=dim, kernel_size=kernel_size),
            PointWiseConv(dim=dim),
        )

        MixerLayer_list = [
            MixerLayer for _ in range(depth)
        ]

        self.MixerLayer_list = nn.Sequential(*MixerLayer_list)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(),
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.bn(x)

        x = self.MixerLayer_list(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


# model = ConvMixer(384, 2).cuda()
# print(model)
# torchsummary.summary(model, (3, 224, 224))
