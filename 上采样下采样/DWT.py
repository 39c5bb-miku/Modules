import torch
import torch.nn as nn
from torch_dwt.functional import dwt3
import numpy as np

# Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation
# https://github.com/apple1986/HWD
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv3d(in_ch*8, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm3d(out_ch),   
                                    nn.ReLU(inplace=True),                                 
                                    ) 
    def forward(self, x):
        coefs = dwt3(x,"haar")
        LLL = coefs[:,0,::]
        LLH = coefs[:,1,::]
        LHL = coefs[:,2,::]
        LHH = coefs[:,3,::]
        HLL = coefs[:,4,::]
        HLH = coefs[:,5,::]
        HHL = coefs[:,6,::]
        HHH = coefs[:,7,::]
        x = torch.cat([LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH], dim=1)        
        x = self.conv_bn_relu(x)

        return x

if __name__ == '__main__':
    with torch.no_grad():
        a = np.load(r'datasets\200-20\train\images\0.npy')
        a = torch.from_numpy(a).reshape(1,1,128,128,128)
        # x = torch.randn(1, 64, 64, 64, 64)
        model = Down_wt(1,1)
        output = model(a)
        output = output.numpy()
        np.save('0',output)
        print(output.shape)