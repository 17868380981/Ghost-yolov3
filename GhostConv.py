import torch
import torch.nn as nn
import math


class GhostConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, fmap_order=None):
        super(GhostConv2d, self).__init__()
        self.fmap_order = fmap_order
        self.oup = out_channels
        init_channels = int(math.ceil(out_channels / ratio))
        new_channels = init_channels*(ratio-1)

        # 本征卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # cheap operation
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        if isinstance(self.fmap_order, list):
            out_sort = out.clone()
            for i, order in enumerate(self.fmap_order):
                out_sort[:, order, :, :] = out[:, i, :, :]   # eg. fmap_order=[3, 0, 1, 2],  0->3, 1->0, 2->1, 3->2
            out = out_sort
        # print(out.size())
        return out[:, :self.oup, :, :]
