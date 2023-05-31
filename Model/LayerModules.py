import torch.nn as nn


class DepthWiseConv(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super(DepthWiseConv, self).__init__()

        self.SpatialMixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(kernel_size, kernel_size), groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.SpatialMixing(x) + x


class PointWiseConv(nn.Module):
    def __init__(self, dim):
        super(PointWiseConv, self).__init__()
        self.ChannelWiseMixing = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.ChannelWiseMixing(x)