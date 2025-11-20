import torch, torch.nn as nn

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
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d_conv1 = self.conv_pair(1024, 512)
        self.d_conv2 = self.conv_pair(512, 256)
        self.d_conv3 = self.conv_pair(256, 128)
        self.d_conv4 = self.conv_pair(128, 64)
        self.o_conv = self.out(64)
    
    def conv_pair(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )
    
    def out(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.conv1(x)
        self.skip1 = x
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        self.skip2 = x
        x = nn.MaxPool2d(2)(x)
        x = self.conv3(x)
        self.skip3 = x
        x = nn.MaxPool2d(2)(x)
        x = self.conv4(x)
        self.skip4 = x
        x = nn.MaxPool2d(2)(x)
        return x
    
    def bottleneck(self, x):
        x = self.b_conv(x)
        return x

    def decode(self, x):
        x = self.t_conv1(x)
        self.skip4 = nn.functional.interpolate(self.skip4, size=x.shape[2:])
        x = torch.cat((self.skip4, x), dim=1)
        x = self.d_conv1(x)

        x = self.t_conv2(x)
        self.skip3 = nn.functional.interpolate(self.skip3, size=x.shape[2:])
        x = torch.cat((self.skip3, x), dim=1)
        x = self.d_conv2(x)

        x = self.t_conv3(x)
        self.skip2 = nn.functional.interpolate(self.skip2, size=x.shape[2:])
        x = torch.cat((self.skip2, x), dim=1)
        x = self.d_conv3(x)
        
        x = self.t_conv4(x)
        self.skip1 = nn.functional.interpolate(self.skip1, size=x.shape[2:])
        x = torch.cat((self.skip1, x), dim=1)
        x = self.d_conv4(x)
        return x

    def shrink(self, x):
        x = self.o_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x)
        x = self.shrink(x)
        return x