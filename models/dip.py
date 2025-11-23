import torch, torch.nn as nn
# import torch.nn.functional as F
import math

class UNet(nn.Module):
	def __init__(self, height, width, out_channels=3, base_channels=32, z_channels=32):
		super(UNet, self).__init__()
		self.out_h = height
		self.out_w = width
		self.out_c = out_channels
		self.bc = base_channels
		self.zc = z_channels

		# ensure divisible by 16 to fit architecture
		pad_h = math.ceil(height / 16) * 16
		pad_w = math.ceil(width / 16) * 16
		
		self.register_buffer('z', torch.randn((1, self.zc, pad_h, pad_w)) * 0.1)

		print(f"DIP initialized with input shape: ({self.zc}, {self.out_h}, {self.out_w}) -> Padded: ({pad_h}, {pad_w})")
		
		self.down1 = self.conv(self.zc, self.bc)
		self.down2 = self.conv(self.bc, self.bc * 2)
		self.down3 = self.conv(self.bc * 2, self.bc * 4)
		self.down4 = self.conv(self.bc * 4, self.bc * 8)

		self.bottleneck = self.conv(self.bc * 8, self.bc * 16)

		self.upscale4 = self.up_conv(self.bc * 16, self.bc * 8)
		self.up4 = self.conv(self.bc * 16, self.bc * 8)

		self.upscale3 = self.up_conv(self.bc * 8, self.bc * 4)
		self.up3 = self.conv(self.bc * 8, self.bc * 4)
		
		self.upscale2 = self.up_conv(self.bc * 4, self.bc * 2)
		self.up2 = self.conv(self.bc * 4, self.bc * 2)

		self.upscale1 = self.up_conv(self.bc * 2, self.bc)
		self.up1 = self.conv(self.bc * 2, self.bc)

		self.out = self.out_conv(self.bc)
	
	def conv(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size, stride, padding, padding_mode='reflect'),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(c_out, c_out, kernel_size, stride, padding, padding_mode='reflect'),
			nn.LeakyReLU(0.2, inplace=True)
		)
	
	def up_conv(self, c_in, c_out):
		return nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
			nn.LeakyReLU(0.2, inplace=True)
		)
	
	def out_conv(self, in_c):
		return nn.Sequential(
			nn.Conv2d(in_c, self.out_c, kernel_size=1),
			nn.Sigmoid()
		)

	# return skip connections
	def down(self, x):
		x = self.down1(x)
		s1 = x
		x = nn.MaxPool2d(2, 2)(x)

		x = self.down2(x)
		s2 = x
		x = nn.MaxPool2d(2, 2)(x)

		x = self.down3(x)
		s3 = x
		x = nn.MaxPool2d(2, 2)(x)

		x = self.down4(x)
		s4 = x
		x = nn.MaxPool2d(2, 2)(x)

		return x, s1, s2, s3, s4

	# take in skip connections
	def up(self, x, s1, s2, s3, s4):
		x = self.upscale4(x)
		x = torch.cat((s4, x), dim=1)
		x = self.up4(x)

		x = self.upscale3(x)
		x = torch.cat((s3, x), dim=1)
		x = self.up3(x)

		x = self.upscale2(x)
		x = torch.cat((s2, x), dim=1)
		x = self.up2(x)

		x = self.upscale1(x)
		x = torch.cat((s1, x), dim=1)
		x = self.up1(x)

		return x

	def forward(self, z):
		x, s1, s2, s3, s4 = self.down(z)
		x = self.bottleneck(x)
		x = self.up(x, s1, s2, s3, s4)
		x = self.out(x)

		# crop back to original HR size
		x = x[:, :, :self.out_h, :self.out_w]
		return x