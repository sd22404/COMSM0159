import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.utils as utils

class SineLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
		super().__init__()
		self.omega_0 = nn.Parameter(torch.tensor(omega_0))
		self.is_first = is_first
		self.in_features = in_features
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		self.init_weights()

	def init_weights(self):
		omega = self.omega_0.item()
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
			else:
				self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega, 
											np.sqrt(6 / self.in_features) / omega)

	def forward(self, input):
		return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

	def forward(self, x):
		res = x
		out = self.conv1(x)
		out = self.relu(out)
		out = self.conv2(out)
		out = out * 0.1 + res
		return out

class RDB_Conv(nn.Module):
	def __init__(self, in_c, out_c, kernel_size=3):
		super(RDB_Conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		out = self.conv(x)
		return torch.cat((x, out), dim=1)

class RDB(nn.Module):
	def __init__(self, in_c, growth, convs, kernel_size=3, res_scale=0.1):
		super(RDB, self).__init__()
		
		self.res_scale = res_scale

		conv_layers = []
		for c in range(convs):
			conv_layers.append(RDB_Conv(in_c + c * growth, growth, kernel_size))
		self.dense_layers = nn.Sequential(*conv_layers)
		
		self.LFF = nn.Conv2d(in_c + convs * growth, in_c, 1, stride=1, padding=0)
		
	def forward(self, x):
		return self.LFF(self.dense_layers(x)) * self.res_scale + x

class RDN(nn.Module):
	def __init__(self, in_channels=3, base_growth=64 // 2, kernel_size=3):
		super(RDN, self).__init__()

		# config
		self.blocks = 20 // 4
		convs = 6 // 2
		growth = 64 // 2
		self.res_scale = 0.1
		
		self.shallow1 = nn.Conv2d(in_channels, base_growth, kernel_size, padding=1)
		self.shallow2 = nn.Conv2d(base_growth, base_growth, kernel_size, padding=1)
		
		RDBs = []
		for _ in range(self.blocks):
			RDBs.append(RDB(base_growth, growth, convs, kernel_size, self.res_scale))
		self.RDBs = nn.Sequential(*RDBs)
		
		self.GFF = nn.Sequential(
			nn.Conv2d(self.blocks * base_growth, base_growth, 1, stride=1, padding=0), # compress
			nn.Conv2d(base_growth, base_growth, kernel_size, stride=1, padding=1) # smooth
		)

		self.out_dim = base_growth 

	def forward(self, x):
		feat1 = self.shallow1(x)
		x = self.shallow2(feat1)

		RDBs_out = []
		current_feat = x
		
		for rdb in self.RDBs:
			current_feat = rdb(current_feat)
			RDBs_out.append(current_feat)

		x = torch.cat(RDBs_out, dim=1)
		x = self.GFF(x)
		x = x * self.res_scale + feat1
		return x


class EDSR(nn.Module):
	def __init__(self, in_channels=3, feat_dim=64, num_res_blocks=16):
		super().__init__()

		self.input = nn.Conv2d(in_channels, feat_dim, 3, padding=1)
		self.res_blocks = nn.Sequential(
			*[ResBlock(feat_dim) for _ in range(num_res_blocks)]
		)
		self.output = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)

	def forward(self, x):
		x = self.input(x)
		res = x
		x = self.res_blocks(x)
		x = self.output(x)
		out = x + res
		return out


class LIIF(nn.Module):
	def __init__(self, in_channels=3, feat_dim=64, mlp_dim=256):
		super().__init__()

		self.encoder = EDSR(in_channels=in_channels, feat_dim=feat_dim, num_res_blocks=8)

		in_dim = 9 * feat_dim + 2 + 2 # 3x3 patch features + relative coord + cell size

		self.siren = nn.Sequential(
			SineLayer(in_dim, mlp_dim, is_first=True),
			SineLayer(mlp_dim, mlp_dim),
			SineLayer(mlp_dim, mlp_dim),
			SineLayer(mlp_dim, mlp_dim),
			# SineLayer(mlp_dim, mlp_dim),
			nn.Linear(mlp_dim, 3),
			nn.Sigmoid()
		)

	def forward(self, lrs, coords, cells):
		B, N, _ = coords.shape
		feats = self.encoder(lrs) # (B, C, H, W)
		B, C, H, W = feats.shape

		# unfold 3x3 patches for each pixel
		feats = F.unfold(feats, 3, padding=1).view(B, C * 9, H, W)

		feat_coords = utils.make_grid(H, W).to(coords.device).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

		# # corners of lr cell
		# corner_offsets = [
		# 	( 1,  1), # bottom right
		# 	( 1, -1), # top right
		# 	(-1,  1), # bottom left
		# 	(-1, -1), # top left
		# ]
		
		dxs = [-1, 1]
		dys = [-1, 1]
		e = 1e-6

		# inputs = []
		areas = []
		preds = []

		# calculate query from neighboring pixels

		# for dx, dy in corner_offsets:
		for dx in dxs:
			for dy in dys:
				coords_ = coords.clone()
				coords_[..., 0] += dy / H + e # (y, x)
				coords_[..., 1] += dx / W + e
				coords_ = torch.clamp(coords_, -1.0 + e, 1.0 - e)
				
				# (B, C, H, W) -> (B, C, N, 1)
				feat_sample = F.grid_sample(
					feats, 
					coords_.flip(-1).unsqueeze(1), # flip (y, x) to (x, y)
					mode='nearest',
					align_corners=False
				).squeeze(2).permute(0, 2, 1)

				coord_sample = F.grid_sample(
					feat_coords, 
					coords_.flip(-1).unsqueeze(1), # flip (y, x) to (x, y)
					mode='nearest',
					align_corners=False
				).squeeze(2).permute(0, 2, 1)

				rel_coord = coords - coord_sample
				rel_coord[:, :, 0] *= H
				rel_coord[:, :, 1] *= W

				rel_cell = cells.clone()
				rel_cell[:, :, 0] *= H
				rel_cell[:, :, 1] *= W

				area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
				areas.append(area + 1e-9)

				input = torch.cat([feat_sample, rel_coord, rel_cell], dim=-1)

				pred = self.siren(input.view(B * N, -1)).view(B, N, -1) # (B, N, 3)
				preds.append(pred)

				# inputs.append(input)
		
		# # stack four corners' inputs
		# stack_input = torch.stack(inputs, dim=0)
		# K, B, N, D = stack_input.shape # K = 4 corners

		# pred = self.siren(stack_input.view(-1, D)).view(K, B, N, -1) # (K, B, N, 3)
		
		total_area = torch.stack(areas, dim=0).sum(dim=0)
		areas[0], areas[3] = areas[3], areas[0] # swap to match preds
		areas[1], areas[2] = areas[2], areas[1]

		rgb = 0
		# for k in range(K):
			# rgb += pred[k] * (areas[k] / total_area).unsqueeze(-1)
		for pred, area in zip(preds, areas):
			rgb += pred * (area / total_area).unsqueeze(-1)
		
		return rgb
