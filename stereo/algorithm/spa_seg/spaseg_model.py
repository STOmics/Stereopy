import torch.nn as nn
import torch.nn.init

use_cuda = torch.cuda.is_available()

# SpaSEG spaseg
class SegNet(nn.Module):
    def __init__(self, input_dim, nChannel, output_dim, nConv):
        super(SegNet, self).__init__()
        self.nConv = nConv
        modules = []
        modules.append(nn.BatchNorm2d(input_dim))
        modules.append(nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1))
        modules.append(nn.BatchNorm2d(nChannel))
        modules.append(nn.LeakyReLU(0.2))

        for i in range(nConv - 1):
            modules.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm2d(nChannel))
            modules.append(nn.LeakyReLU(0.2))

        modules.append(nn.Conv2d(nChannel, output_dim, kernel_size=1, stride=1, padding=0))
        modules.append(nn.BatchNorm2d(output_dim))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)
