import torch.nn as nn

class DIP(nn.Module):
    def __init__(self, height, width, channels):
        super(DIP, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        
        self.conv1 = self.conv_pair(self.channels, 64)
        self.conv2 = self.conv_pair(64, 128)
        self.conv3 = self.conv_pair(128, 256)
        self.conv4 = self.conv_pair(256, 512)
        self.b_conv = self.conv_pair(512, 1024)
        self.t_conv1 = self.t_conv_pair(1024, 512)
        self.t_conv2 = self.t_conv_pair(512, 256)
        self.t_conv3 = self.t_conv_pair(256, 128)
        self.t_conv4 = self.t_conv_pair(128, 64)
        self.s_conv = nn.Conv2d(64, self.channels, kernel_size=1)
    
    def conv_pair(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def t_conv_pair(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            self.conv_pair(in_channels, out_channels, kernel_size, stride, padding)
        )

    def encode(self, x):
        x = self.conv1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv3(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(2)(x)
        return x
    
    def bottleneck(self, x):
        x = self.b_conv(x)
        return x

    def decode(self, x):
        x = self.t_conv1(x)
        x = self.t_conv2(x)
        x = self.t_conv3(x)
        x = self.t_conv4(x)
        return x

    def shrink(self, x):
        x = self.s_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x)
        x = self.shrink(x)
        return x