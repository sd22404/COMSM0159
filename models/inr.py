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

		in_dim = 9 * feat_dim + 2 #+ 2

		self.siren = nn.Sequential(
			SineLayer(in_dim, mlp_dim, is_first=True),
			SineLayer(mlp_dim, mlp_dim),
			SineLayer(mlp_dim, mlp_dim),
			nn.Linear(mlp_dim, 3),
			nn.Sigmoid()
		)

	def forward(self, lrs, coords, cells):
		B, N, _ = coords.shape
		
		feats = self.encoder(lrs) # (B, C, H, W)
		B, C, H, W = feats.shape

		# unfold 3x3 patches for each pixel
		feats = F.unfold(feats, 3, padding=1).view(B, C * 9, H, W)
		
		# corners of lr cell
		dxs = [-1, 1]
		dys = [-1, 1]
		e = 1e-6

		def make_coord(shape, ranges=None, flatten=True):
			""" Make coordinates at grid centers.
			"""
			coord_seqs = []
			for i, n in enumerate(shape):
				if ranges is None:
					v0, v1 = -1, 1
				else:
					v0, v1 = ranges[i]
				r = (v1 - v0) / (2 * n)
				seq = v0 + r + (2 * r) * torch.arange(n).float()
				coord_seqs.append(seq)
			ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
			# (y, x) -> (x, y)
			ret = torch.stack([ret[..., 1], ret[..., 0]], dim=-1)
			if flatten:
				ret = ret.view(-1, ret.shape[-1])
			return ret

		feat_coords = make_coord((H, W), flatten=False).to(coords.device).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

		preds = []
		areas = []

		# calculate query from neighboring pixels
		for dx in dxs:
			for dy in dys:
				coords_ = coords.clone()
				coords_[..., 0] += dx / W + e
				coords_[..., 1] += dy / H + e
				coords_ = torch.clamp(coords_, -1.0 + e, 1.0 - e)
				
				# (B, C, H, W) -> (B, C, N, 1)
				feat_sample = F.grid_sample(
					feats, 
					coords_.unsqueeze(1),
					mode='bicubic',
					align_corners=False
				).squeeze(2).permute(0, 2, 1)

				coord_sample = F.grid_sample(
					feat_coords, 
					coords_.unsqueeze(1),
					mode='bicubic',
					align_corners=False
				).squeeze(2).permute(0, 2, 1)

				rel_coord = coords - coord_sample
				rel_coord[:, :, 0] *= H
				rel_coord[:, :, 1] *= W

				# append to feats
				input = torch.cat([feat_sample, rel_coord], dim=-1)

				# rel_cell = cells.clone()
				# rel_cell[:, :, 0] *= feats.shape[-2]
				# rel_cell[:, :, 1] *= feats.shape[-1]
				# input = torch.cat([input, rel_cell], dim=-1)

				pred = self.siren(input.view(B * N, -1)).view(B, N, -1) # (B, N, 3)
				preds.append(pred)

				area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
				areas.append(area + 1e-9)
		
		total_area = torch.stack(areas, dim=0).sum(dim=0)
		t = areas[0]; areas[0] = areas[3]; areas[3] = t  # swap areas to match preds order
		t = areas[1]; areas[1] = areas[2]; areas[2] = t

		rgb = 0
		for pred, area in zip(preds, areas):
			rgb += pred * (area / total_area).unsqueeze(-1)
		
		return rgb
