import torch
import torch.nn as nn
import torch.nn.functional as F


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
	def __init__(self, in_channels=3, base_growth=32, kernel_size=3):
		super(RDN, self).__init__()

		# config
		self.blocks = 12 #20
		convs = 4 #6
		growth = 32 #64
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


class LIIF(nn.Module):
	def __init__(self, in_channels=3, feat_dim=64, mlp_dim=256):
		super().__init__()

		self.encoder = RDN(in_channels=in_channels, base_growth=feat_dim)

		in_dim = 9 * feat_dim + 2 + 2 # 3x3 patch features + relative coord + cell size

		self.mlp = nn.Sequential(
			nn.Linear(in_dim, mlp_dim),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_dim, mlp_dim),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_dim, mlp_dim),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_dim, mlp_dim),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_dim, 3),
		)

	def forward(self, lrs, coords, cells):
		B, N, _ = coords.shape
		feats = self.encoder(lrs) # (B, C, H, W)
		B, C, H, W = feats.shape

		# unfold 3x3 patches for each pixel
		feats = F.unfold(feats, 3, padding=1).view(B, C * 9, H, W)

		# pixel coordinates from 0 to H-1 / W-1
		coords_px = (coords + 1.0) * 0.5
		coords_px[..., 0] = coords_px[..., 0] * H - 0.5
		coords_px[..., 1] = coords_px[..., 1] * W - 0.5

		# pixel space corners
		y0 = torch.floor(coords_px[..., 0]).clamp(0, H - 1)
		x0 = torch.floor(coords_px[..., 1]).clamp(0, W - 1)
		y1 = (y0 + 1).clamp(0, H - 1)
		x1 = (x0 + 1).clamp(0, W - 1)

		def to_norm(y, x):
			y = (y + 0.5) * 2 / H - 1
			x = (x + 0.5) * 2 / W - 1
			return torch.stack([y, x], dim=-1)

		corner_coords = [
			to_norm(y0, x0), # top left
			to_norm(y0, x1), # top right
			to_norm(y1, x0), # bottom left
			to_norm(y1, x1), # bottom right
		]

		ty = coords_px[..., 0] - y0
		tx = coords_px[..., 1] - x0
		corner_weights = [
			(1 - ty) * (1 - tx),
			(1 - ty) * tx,
			ty * (1 - tx),
			ty * tx,
		]

		rel_cell = cells.clone()
		rel_cell[:, :, 0] *= H
		rel_cell[:, :, 1] *= W

		inputs = []
		for corner_coord in corner_coords:
			grid = corner_coord.flip(-1).unsqueeze(1) # (y, x) -> (x, y)
			feat_sample = F.grid_sample(
				feats,
				grid,
				mode='nearest',
				align_corners=False
			).squeeze(2).permute(0, 2, 1)

			rel_coord = coords - corner_coord
			rel_coord[:, :, 0] *= H
			rel_coord[:, :, 1] *= W

			input = torch.cat([feat_sample, rel_coord, rel_cell], dim=-1)
			inputs.append(input)
		
		# (B, 4 * N, D)
		inputs = torch.cat(inputs, dim=1)
		preds = self.mlp(inputs.view(B * 4 * N, -1)).view(B, 4 * N, 3)

		preds = preds.chunk(4, dim=1)
		
		# blend predictions from 4 corners
		rgb = sum(pred * weight.unsqueeze(-1) for pred, weight in zip(preds, corner_weights))
		
		return rgb
