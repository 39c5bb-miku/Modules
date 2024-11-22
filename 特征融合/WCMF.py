import torch
import torch.nn as nn

# FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
# https://github.com/XiaoJinNK/FCMNet
class WCMF(nn.Module):
    def __init__(self,channel=256):
        super(WCMF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_d1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

        self.conv_c1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def fusion(self,f1,f2,f_vec):

        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2
    def forward(self,rgb,depth):
        Fr = self.conv_r1(rgb)
        Fd = self.conv_d1(depth)
        f = torch.cat([Fr, Fd],dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        # f = self.avgpool(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    residual = torch.randn(2, 64, 32, 32)
    model = WCMF(64)
    output = model(x, residual)
    print(output.shape)