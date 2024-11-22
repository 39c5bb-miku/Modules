import torch
import torch.nn as nn
import torch.nn.functional as F

# Comprehensive and Delicate: An Efficient Transformer for Image Restoration(CVPR 2023)
# https://github.com/XLearning-SCU/2023-CVPR-CODE
class DualAdaptiveNeuralBlock(nn.Module):
    def __init__(self, embed_dim):
        super(DualAdaptiveNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        self.group_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Conv2d(embed_dim, embed_dim * 2, 7, 1, 3, groups=embed_dim)
        )
        self.post_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x0, x1 = self.group_conv(x).view(B, C, 2, H, W).chunk(2, dim=2)
        x_ = F.gelu(x0.squeeze(2)) * torch.sigmoid(x1.squeeze(2))
        x_ = self.post_conv(x_)
        return x_

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    model = DualAdaptiveNeuralBlock(64)
    output = model(x)
    print(output.shape)