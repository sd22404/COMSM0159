import torch, torch.nn as nn
import torch.nn.functional as F

class DIP(nn.Module):
	def __init__(self, height, width, channels):
		super(DIP, self).__init__()
		self.height = height
		self.width = width
		self.channels = channels
		self.register_buffer('z', torch.randn((1, self.channels, self.height, self.width)))
		self.b = 32

		print(f"DIP initialized with input shape: ({self.channels}, {self.height}, {self.width})")
		
		self.conv1 = self.conv_pair(self.channels, self.b)
		self.conv2 = self.conv_pair(self.b, self.b * 2)
		self.conv3 = self.conv_pair(self.b * 2, self.b * 4)
		self.conv4 = self.conv_pair(self.b * 4, self.b * 8)

		self.b_conv = self.conv_pair(self.b * 8, self.b * 16)
		# self.b_conv = self.conv_pair(self.b * 4, self.b * 8)

		self.t_conv4 = nn.ConvTranspose2d(self.b * 16, self.b * 8, kernel_size=2, stride=2, padding=0)
		self.d_conv4 = self.conv_pair(self.b * 16, self.b * 8)

		self.t_conv3 = nn.ConvTranspose2d(self.b * 8, self.b * 4, kernel_size=2, stride=2, padding=0)
		self.d_conv3 = self.conv_pair(self.b * 8, self.b * 4)
		
		self.t_conv2 = nn.ConvTranspose2d(self.b * 4, self.b * 2, kernel_size=2, stride=2, padding=0)
		self.d_conv2 = self.conv_pair(self.b * 4, self.b * 2)

		self.t_conv1 = nn.ConvTranspose2d(self.b * 2, self.b, kernel_size=2, stride=2, padding=0)
		self.d_conv1 = self.conv_pair(self.b * 2, self.b)

		self.o_conv = self.out(self.b)
	
	def conv_pair(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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
		s1 = x
		x = nn.MaxPool2d(2)(x)

		x = self.conv2(x)
		s2 = x
		x = nn.MaxPool2d(2)(x)

		x = self.conv3(x)
		s3 = x
		x = nn.MaxPool2d(2)(x)

		x = self.conv4(x)
		s4 = x
		x = nn.MaxPool2d(2)(x)

		return x, s1, s2, s3, s4
	
	def bottleneck(self, x):
		x = self.b_conv(x)
		return x

	def decode(self, x, s1, s2, s3, s4):
		x = self.t_conv4(x)
		if x.shape[2:] != s4.shape[2:]:
			x = F.interpolate(x, size=s4.shape[2:], mode='bicubic')
		x = torch.cat((s4, x), dim=1)
		x = self.d_conv4(x)

		x = self.t_conv3(x)
		if x.shape[2:] != s3.shape[2:]:
			x = F.interpolate(x, size=s3.shape[2:], mode='bicubic')
		x = torch.cat((s3, x), dim=1)
		x = self.d_conv3(x)

		x = self.t_conv2(x)
		if x.shape[2:] != s2.shape[2:]:
			x = F.interpolate(x, size=s2.shape[2:], mode='bicubic')
		x = torch.cat((s2, x), dim=1)
		x = self.d_conv2(x)

		x = self.t_conv1(x)
		if x.shape[2:] != s1.shape[2:]:
			x = F.interpolate(x, size=s1.shape[2:], mode='bicubic')
		x = torch.cat((s1, x), dim=1)
		x = self.d_conv1(x)

		return x

	def shrink(self, x):
		x = self.o_conv(x)
		return x

	def forward(self, z):
		x, s1, s2, s3, s4 = self.encode(z)
		x = self.bottleneck(x)
		x = self.decode(x, s1, s2, s3, s4)
		x = self.shrink(x)
		return x