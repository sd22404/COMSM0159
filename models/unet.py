import torch, torch.nn as nn
import math

class UNet(nn.Module):
	def __init__(self, height, width, out_channels=3, base_channels=32, z_channels=32, z_sigma=0.05):
		super(UNet, self).__init__()
		self.out_h = height
		self.out_w = width
		self.out_c = out_channels
		self.bc = base_channels
		self.zc = z_channels
		self.std = z_sigma

		# ensure divisible by 16 to fit architecture
		pad_h = math.ceil(height / 16) * 16
		pad_w = math.ceil(width / 16) * 16
		
		self.register_buffer('z', torch.randn((1, self.zc, pad_h, pad_w)) * 0.1)

		print(f"DIP initialized with input shape: ({self.zc}, {self.out_h}, {self.out_w}) -> Padded: ({pad_h}, {pad_w})")
		
		self.input = conv(self.zc, self.bc)

		self.down1 = down_conv(self.bc, self.bc * 2)
		self.down2 = down_conv(self.bc * 2, self.bc * 4)
		self.down3 = down_conv(self.bc * 4, self.bc * 8)
		self.down4 = down_conv(self.bc * 8, self.bc * 16)

		self.up4 = up_conv(self.bc * 16, self.bc * 8)
		self.up3 = up_conv(self.bc * 8, self.bc * 4)
		self.up2 = up_conv(self.bc * 4, self.bc * 2)
		self.up1 = up_conv(self.bc * 2, self.bc)

		self.output = out_conv(self.bc, self.out_c)

	def get_z(self, sigma=0.0):
		if sigma <= 0:
			return self.z
		return self.z + sigma * torch.randn_like(self.z)

	def forward(self, z=None):
		z = self.get_z(self.std) if z is None else z
		x = self.input(z)

		s1 = x
		x = self.down1(x)
		s2 = x
		x = self.down2(x)
		s3 = x
		x = self.down3(x)
		s4 = x
		x = self.down4(x)

		x = self.up4(x, s4)
		x = self.up3(x, s3)
		x = self.up2(x, s2)
		x = self.up1(x, s1)

		x = self.output(x)

		# crop back to original HR size
		x = x[:, :, :self.out_h, :self.out_w]
		return x


class conv(nn.Module):
	def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
		super(conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride, padding, padding_mode='reflect'),
		)

	def forward(self, x):
		return self.conv(x)

class double_conv(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			conv(c_in, c_out, kernel_size, stride, padding),
			nn.LeakyReLU(0.2, inplace=True),
			conv(c_out, c_out, kernel_size, stride, padding),
			nn.LeakyReLU(0.2, inplace=True)
		)
	
	def forward(self, x):
		return self.conv(x)

class down_conv(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		super(down_conv, self).__init__()
		self.down = nn.Sequential(
			# nn.MaxPool2d(2, 2),
			conv(c_in, c_out, kernel_size=2, stride=2, padding=0),
			double_conv(c_out, c_out, kernel_size, stride, padding)
		)

	def forward(self, x):
		return self.down(x)
	
class up_conv(nn.Module):
	def __init__(self, c_in, c_out):
		super(up_conv, self).__init__()
		# self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
		)
		self.conv = double_conv(c_in, c_out)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat((x, skip), dim=1)
		return self.conv(x)

class out_conv(nn.Module):
	def __init__(self, in_c, out_c):
		super(out_conv, self).__init__()
		self.conv = nn.Sequential(
			conv(in_c, out_c, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.conv(x)