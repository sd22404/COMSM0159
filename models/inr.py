import math
import torch
import torch.nn as nn


class MultiResolutionHashEncoder(nn.Module):
	def __init__(
		self,
		num_levels=12,
		features_per_level=8,
		base_resolution=16,
		finest_resolution=512,
		hashmap_size=2 ** 17,
	):
		super().__init__()
		self.num_levels = num_levels
		self.features_per_level = features_per_level
		if num_levels > 1:
			per_level_scale = math.exp(
				(math.log(finest_resolution) - math.log(base_resolution)) / (num_levels - 1)
			)
		else:
			per_level_scale = 1.0
		self.resolutions = [
			max(2, int(base_resolution * (per_level_scale ** level))) for level in range(num_levels)
		]
		self.hash_sizes = [min(hashmap_size, res * res) for res in self.resolutions]
		self.offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
		self.primes = torch.tensor([1, 2654435761], dtype=torch.long)
		self.embeddings = nn.ModuleList(
			[
				nn.Embedding(hash_size, features_per_level)
				for hash_size in self.hash_sizes
			]
		)
		for table in self.embeddings:
			nn.init.uniform_(table.weight, -1e-4, 1e-4)

	def forward(self, coords):
		if coords.dim() == 2:
			coords = coords.unsqueeze(0)
		coords = coords.clamp(0.0, 1.0)
		B, N, _ = coords.shape
		offsets = self.offsets.to(coords.device)
		primes = self.primes.to(coords.device)
		level_features = []
		for res, table, hash_size in zip(self.resolutions, self.embeddings, self.hash_sizes):
			scaled = coords * res
			base_index = torch.floor(scaled).long()
			frac = scaled - base_index.float()
			corner_index = base_index.unsqueeze(-2) + offsets.view(1, 1, 4, 2)
			corner_index = corner_index.clamp(0, res - 1)
			hash_codes = (corner_index[..., 0] * primes[0]) ^ (corner_index[..., 1] * primes[1])
			hash_codes = torch.remainder(hash_codes, hash_size)
			embedded = table(hash_codes)
			dx = frac[..., 0:1]
			dy = frac[..., 1:2]
			weights = torch.stack(
				[
					(1.0 - dx) * (1.0 - dy),
					(1.0 - dx) * dy,
					dx * (1.0 - dy),
					dx * dy,
				],
				dim=-2,
			)
			feature = (embedded * weights).sum(dim=-2)
			level_features.append(feature)
		encoded = torch.cat(level_features, dim=-1)
		return encoded, level_features


class SelfAttentionBlock(nn.Module):
	def __init__(self, dim, heads=4, dropout=0.0):
		super().__init__()
		self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
		self.norm1 = nn.LayerNorm(dim)
		self.ff = nn.Sequential(
			nn.Linear(dim, dim * 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim * 2, dim),
			nn.Dropout(dropout),
		)
		self.norm2 = nn.LayerNorm(dim)

	def forward(self, x, context=None):
		if context is None:
			attn_out, _ = self.attn(x, x, x)
		else:
			attn_out, _ = self.attn(x, context, context)
		x = self.norm1(x + attn_out)
		return self.norm2(x + self.ff(x))


class TopDownSelfAttention(nn.Module):
	def __init__(self, num_levels, dim, heads=4, dropout=0.0):
		super().__init__()
		self.num_levels = num_levels
		self.blocks = nn.ModuleList(
			[SelfAttentionBlock(dim, heads=heads, dropout=dropout) for _ in range(num_levels)]
		)

	def forward(self, features):
		context = None
		outputs = [None] * self.num_levels
		for level in range(self.num_levels):
			feat = features[level]
			feat = self.blocks[level](feat, context)
			context = feat
			outputs[level] = feat
		return torch.cat(outputs, dim=-1)


class LRFeatureEncoder(nn.Module):
	def __init__(self, in_channels=3, hidden_dim=128, latent_dim=256):
		super().__init__()
		layers = []
		channels = in_channels
		for idx in range(3):
			out_channels = hidden_dim * (2 ** idx)
			layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			layers.append(nn.BatchNorm2d(out_channels))
			channels = out_channels
		self.features = nn.Sequential(*layers)
		self.proj = nn.Linear(channels, latent_dim)

	def forward(self, x):
		h = self.features(x)
		h = h.mean(dim=[2, 3])
		return self.proj(h)


class ImplicitMLP(nn.Module):
	def __init__(self, input_dim, hidden_dim=256, out_dim=3, num_layers=4, dropout=0.0):
		super().__init__()
		layers = []
		dim = input_dim
		for _ in range(num_layers):
			layers.append(nn.Linear(dim, hidden_dim))
			layers.append(nn.GELU())
			layers.append(nn.Dropout(dropout))
			dim = hidden_dim
		self.net = nn.Sequential(*layers)
		self.out = nn.Linear(hidden_dim, out_dim)

	def forward(self, x):
		return self.out(self.net(x))


class SRINR(nn.Module):
	def __init__(
		self,
		in_channels=3,
		out_channels=3,
		num_levels=12,
		features_per_level=8,
		base_resolution=16,
		finest_resolution=512,
		hashmap_size=2 ** 17,
		hidden_dim=256,
		num_mlp_layers=5,
		heads=4,
		dropout=0.0,
		output_activation="sigmoid",
	):
		super().__init__()
		self.encoder = MultiResolutionHashEncoder(
			num_levels=num_levels,
			features_per_level=features_per_level,
			base_resolution=base_resolution,
			finest_resolution=finest_resolution,
			hashmap_size=hashmap_size,
		)
		self.attn = TopDownSelfAttention(num_levels, features_per_level, heads=heads, dropout=dropout)
		self.lr_encoder = LRFeatureEncoder(in_channels=in_channels, hidden_dim=hidden_dim // 4, latent_dim=hidden_dim)
		attn_dim = num_levels * features_per_level
		self.mlp = ImplicitMLP(
			input_dim=attn_dim + hidden_dim + 2,
			hidden_dim=hidden_dim,
			out_dim=out_channels,
			num_layers=num_mlp_layers,
			dropout=dropout,
		)
		self.output_activation = output_activation

	def forward(self, lr_image, query_coords):
		if query_coords.dim() == 2:
			query_coords = query_coords.unsqueeze(0)
		encoded, level_feats = self.encoder(query_coords)
		attn_feats = self.attn(level_feats)
		lr_latent = self.lr_encoder(lr_image)
		lr_latent = lr_latent.unsqueeze(1).expand(-1, query_coords.shape[1], -1)
		features = torch.cat((query_coords, attn_feats, lr_latent), dim=-1)
		out = self.mlp(features)
		if self.output_activation == "sigmoid":
			out = torch.sigmoid(out)
		elif self.output_activation == "tanh":
			out = torch.tanh(out)
		return out
