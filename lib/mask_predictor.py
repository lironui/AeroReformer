import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from arc import AdaptiveRotatedConv2d, RountingFunction
from lib.linear_attention import ChannelAttention, SpatialAttention
import math


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                # m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                # m.append(nn.PixelShuffle(2))
                m.append(nn.ConvTranspose2d(num_feat, num_feat, kernel_size=2, stride=2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, encoder_channels=(128, 256, 512, 1024)):
        super(SimpleDecoding, self).__init__()

        self.lat1 = self._lateral_block(encoder_channels[0])
        self.lat2 = self._lateral_block(encoder_channels[1])
        self.lat3 = self._lateral_block(encoder_channels[2])
        self.lat4 = self._lateral_block(encoder_channels[3])

        self.up4 = Upsample(scale=2, num_feat=encoder_channels[3])
        self.up3 = Upsample(scale=2, num_feat=encoder_channels[2])
        self.up2 = Upsample(scale=2, num_feat=encoder_channels[1])

        self.smooth1 = self._adaptive_smooth_block(encoder_channels[0])
        self.smooth2 = self._adaptive_smooth_block(encoder_channels[1])
        self.smooth3 = self._adaptive_smooth_block(encoder_channels[2])

        self.reduce4 = self._fusion_block(encoder_channels[3], encoder_channels[2])
        self.reduce3 = self._fusion_block(encoder_channels[2], encoder_channels[1])
        self.reduce2 = self._fusion_block(encoder_channels[1], encoder_channels[0])

        self.ca2 = ChannelAttention()
        self.ca3 = ChannelAttention()
        self.pa1 = SpatialAttention(in_places=encoder_channels[1] + encoder_channels[0])

        # Final 1×1 convolution for mask prediction
        self.final_conv = nn.Conv2d(encoder_channels[0], 2, kernel_size=1)

    def forward(self, x4, x3, x2, x1):
        lat4 = self.lat4(x4)  # [B, 1024, 15, 15]
        x4_up = self.up4(lat4)  # [B, 1024, 30, 30]
        lat3 = self.lat3(x3)  # [B, 512, 30, 30]

        fuse3 = torch.cat([lat3, x4_up], dim=1)  # [B, 1024 + 512, 30, 30]
        fuse3 = self.ca3(fuse3)
        fuse3 = self.reduce4(fuse3)  # [B, 512, 30, 30]
        fuse3 = self.smooth3(fuse3)

        x3_up = self.up3(fuse3)  # [B, 512, 60, 60]
        lat2 = self.lat2(x2)  # [B, 256, 60, 60]
        fuse2 = torch.cat([lat2, x3_up], dim=1)  # [B, 512 + 256, 60, 60]
        fuse2 = self.ca2(fuse2)
        fuse2 = self.reduce3(fuse2)  # [B, 256, 60, 60]
        fuse2 = self.smooth2(fuse2)

        x2_up = self.up2(fuse2)  # [B, 256, 120, 120]
        lat1 = self.lat1(x1)  # [B, 128, 120, 120]
        fuse1 = torch.cat([lat1, x2_up], dim=1)  # [B, 256 + 128, 120, 120]
        fuse1 = self.pa1(fuse1)
        fuse1 = self.reduce2(fuse1)  # [B, 128, 120, 120]
        fuse1 = self.smooth1(fuse1)

        mask = self.final_conv(fuse1)  # [B, 2, 120, 120]

        return mask

    def _lateral_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _adaptive_smooth_block(self, channels):
        routing_function = RountingFunction(in_channels=channels, kernel_number=1)
        return nn.Sequential(
            AdaptiveRotatedConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                                  padding=1, rounting_func=routing_function, bias=False, kernel_number=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _fusion_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )


def main():
    import torch

    batch_size = 1
    x1 = torch.randn(batch_size, 128, 120, 120)
    x2 = torch.randn(batch_size, 256, 60, 60)
    x3 = torch.randn(batch_size, 512, 30, 30)
    x4 = torch.randn(batch_size, 1024, 15, 15)

    model = SimpleDecoding(c4_dims=1024, encoder_channels=(128, 256, 512, 1024))
    model.eval()

    with torch.no_grad():
        mask = model(x4, x3, x2, x1)

    print("mask shape:", mask.shape)

if __name__ == "__main__":
    main()
