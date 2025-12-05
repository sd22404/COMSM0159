import torch
import torch.nn as nn
import math


class UNet(nn.Module):
	def __init__(self, height, width, out_channels=3, base_channels=128, z_channels=32, skip_channels=4, z_sigma=0.05):
		super(UNet, self).__init__()
		self.out_h = height
		self.out_w = width
		self.out_c = out_channels
		self.bc = base_channels
		self.zc = z_channels
		self.sc = skip_channels
		self.std = z_sigma

		# ensure divisible by 32 to fit architecture
		pad_h = math.ceil(height / 32) * 32
		pad_w = math.ceil(width / 32) * 32
		
		self.register_buffer('z', torch.randn((1, self.zc, pad_h, pad_w)) * 0.1)

		print(f"UNet initialized with input shape: ({self.zc}, {self.out_h}, {self.out_w}) -> Padded: ({pad_h}, {pad_w})")
		
		self.skip1 = skip_block(self.zc, self.sc)
		self.down1 = down_block(self.zc, self.bc)

		self.skip2 = skip_block(self.bc, self.sc)
		self.down2 = down_block(self.bc, self.bc)

		self.skip3 = skip_block(self.bc, self.sc)
		self.down3 = down_block(self.bc, self.bc)

		self.skip4 = skip_block(self.bc, self.sc)
		self.down4 = down_block(self.bc, self.bc)

		self.skip5 = skip_block(self.bc, self.sc)
		self.down5 = down_block(self.bc, self.bc)
		
		self.up5 = up_block(self.bc, self.sc, self.bc)
		self.up4 = up_block(self.bc, self.sc, self.bc)
		self.up3 = up_block(self.bc, self.sc, self.bc)
		self.up2 = up_block(self.bc, self.sc, self.bc)
		self.up1 = up_block(self.bc, self.sc, self.bc)

		self.output = nn.Sequential(
			nn.Conv2d(self.bc, self.out_c, 1),
			nn.Sigmoid()
		)

	def get_z(self, sigma=0.0):
		if sigma <= 0:
			return self.z
		return self.z + sigma * torch.randn_like(self.z)

	def forward(self, z=None):
		z = self.get_z(self.std) if z is None else z

		s1 = self.skip1(z)
		x = self.down1(z)

		s2 = self.skip2(x)
		x = self.down2(x)
		s3 = self.skip3(x)
		x = self.down3(x)

		s4 = self.skip4(x)
		x = self.down4(x)

		s5 = self.skip5(x)
		x = self.down5(x)

		x = self.up5(x, s5)
		x = self.up4(x, s4)
		x = self.up3(x, s3)
		x = self.up2(x, s2)
		x = self.up1(x, s1)

		x = self.output(x)

		# crop back to original HR size
		x = x[:, :, :self.out_h, :self.out_w]
		return x

class skip_block(nn.Module):
	def __init__(self, c_in, c_out):
		super(skip_block, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(c_out),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, x):
		return self.conv(x)

class down_block(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		super(down_block, self).__init__()
		self.down = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size, stride=2, padding=padding),
			nn.BatchNorm2d(c_out),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(c_out, c_out, kernel_size, stride, padding=padding),
			nn.BatchNorm2d(c_out),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, x):
		return self.down(x)

class up_block(nn.Module):
	def __init__(self, c_in, c_skip, c_out, kernel_size=3, padding=1):
		super(up_block, self).__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear')
		self.conv = nn.Sequential(
			nn.BatchNorm2d(c_in + c_skip),
			nn.Conv2d(c_in + c_skip, c_out, kernel_size, stride=1, padding=padding),
			nn.BatchNorm2d(c_out),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(c_out),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat((x, skip), dim=1)
		return self.conv(x)
