import torch
import torch.nn as nn
import torch.nn.functional as F

# Restormer: Efficient Transformer for High-Resolution Image Restoration(CVPR 2022)
# https://github.com/swz30/Restormer
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    model = FeedForward(64, 2.66, False)
    output = model(x)
    print(output.shape)