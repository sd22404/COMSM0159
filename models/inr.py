import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SineLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
		super().__init__()
		self.omega_0 = omega_0
		self.is_first = is_first
		self.in_features = in_features
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		self.init_weights()

	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
			else:
				self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
											np.sqrt(6 / self.in_features) / self.omega_0)

	def forward(self, input):
		return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
		# self.bn1 = nn.BatchNorm2d(channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
		# self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		res = x
		out = self.conv1(x)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		# out = self.bn2(out)
		out += res
		# out = self.relu(out)
		return out

class LIIF(nn.Module):
	def __init__(self, in_channels=3, feat_dim=64, mlp_dim=256):
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, feat_dim, 3, padding=1),
			nn.ReLU(inplace=True),
			ResBlock(feat_dim),
			ResBlock(feat_dim),
			ResBlock(feat_dim),
			ResBlock(feat_dim),
			nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
		)

		in_dim = feat_dim + 2

		self.mlp = nn.Sequential(
			SineLayer(in_dim, mlp_dim, is_first=True),
			SineLayer(mlp_dim, mlp_dim),
			SineLayer(mlp_dim, mlp_dim),
			nn.Linear(mlp_dim, 3),
			nn.Sigmoid()
		)

	def forward(self, lrs, coords):
		B, N, _ = coords.shape
		
		feats = self.encoder(lrs) # (B, C, H, W)

		# convert to [-1, 1]
		grid = coords.view(B, N, 1, 2) * 2 - 1
		
		# (B, C, H, W) -> (B, C, N, 1)
		samples = F.grid_sample(
			feats, 
			grid,
			mode='bicubic', 
			padding_mode='border'
		)
		samples = samples.squeeze(-1).permute(0, 2, 1) # (B, N, C)

		# TODO: IMPLEMENT RELATIVE COORDS

		inp = torch.cat([samples, coords], dim=-1) # (B, N, C+2)

		rgb = self.mlp(inp) # (B, N, 3)
		
		return rgb
