import torch
import torch.nn as nn
import torch.nn.functional as F

# FastICENet: A real-time and accurate semantic segmentation model for aerial remote sensing river ice image
# https://github.com/nwpulab113/FastICENet
class UpSample(nn.Module):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    model = UpSample(64)
    output = model(x)
    print(output.shape)