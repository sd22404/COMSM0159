import math
import torch
import torch.nn as nn


class Diffuser(nn.Module):
	def __init__(self, denoiser, image_resolution, n_times, beta_minmax=(1e-4, 2e-2), device='cuda'):
		super(Diffuser, self).__init__()
		self.model = denoiser
		self.n_times = n_times
		self.img_H, self.img_W, self.img_C = image_resolution
		self.device = torch.device(device)
		beta_1, beta_T = beta_minmax
		betas = torch.linspace(beta_1, beta_T, steps=n_times, device=self.device, dtype=torch.float32)
		alphas = 1.0 - betas
		alpha_bars = torch.cumprod(alphas, dim=0)
		self.register_buffer('betas', betas)
		self.register_buffer('sqrt_betas', torch.sqrt(betas))
		self.register_buffer('alphas', alphas)
		self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
		self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
		self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))

	def to(self, *args, **kwargs):
		module = super(Diffuser, self).to(*args, **kwargs)
		param = next(module.parameters(), None)
		if param is not None:
			module.device = param.device
		else:
			buf = next(module.buffers(), None)
			if buf is not None:
				module.device = buf.device
		return module

	def extract(self, buf, t, x_shape):
		t = t.reshape(-1).long()
		out = buf.gather(0, t.to(buf.device))
		return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

	def scale_to_minus_one_to_one(self, x):
		return x * 2.0 - 1.0

	def reverse_scale_to_zero_to_one(self, x):
		return (x + 1.0) * 0.5

	def add_noise(self, x_zeros, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_zeros)
		sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
		sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
		x_t = x_zeros * sqrt_alpha_bar + noise * sqrt_one_minus_alpha_bar
		return x_t.detach(), noise

	def make_noisy(self, x_zeros, t):
		return self.add_noise(x_zeros, t)

	def forward(self, x_zeros, cond):
		x_zeros = self.scale_to_minus_one_to_one(x_zeros)
		if cond is not None:
			cond = self.scale_to_minus_one_to_one(cond)
		self.device = x_zeros.device
		batch_size = x_zeros.shape[0]
		t = torch.randint(0, self.n_times, (batch_size,), device=self.device)
		perturbed, noise = self.add_noise(x_zeros, t)
		if cond is not None:
			cond = cond.to(self.device)
			if cond.shape[0] != batch_size:
				cond = cond.expand(batch_size, -1, -1, -1)
		pred_noise = self.model(perturbed, t, cond)
		return perturbed, noise, pred_noise

	def denoise_at_t(self, x_t, timestep, cond):
		cond = cond.to(x_t.device)
		if cond.shape[0] != x_t.shape[0]:
			cond = cond.expand(x_t.shape[0], -1, -1, -1)
		cond = self.scale_to_minus_one_to_one(cond)
		# stochastic noise only for t > 0
		z = torch.randn_like(x_t) * (timestep > 0).float().view(-1, 1, 1, 1)
		# predict noise
		eps = self.model(x_t, timestep, cond)
		alpha_t = self.extract(self.alphas, timestep, x_t.shape)
		sqrt_alpha_t = self.extract(self.sqrt_alphas, timestep, x_t.shape)
		sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
		beta_t = 1.0 - alpha_t
		alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, x_t.shape) ** 2
		prev_t = torch.clamp(timestep - 1, min=0)
		alpha_bar_prev = self.extract(self.sqrt_alpha_bars, prev_t, x_t.shape) ** 2
		# DDPM mean
		mean = (x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_t
		posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
		sqrt_posterior_variance = torch.sqrt(torch.clamp(posterior_variance, min=1e-12))
		x_prev = mean + sqrt_posterior_variance * z
		return x_prev.clamp(-1.0, 1.0)

	def sample(self, N, lr_up):
		device = self.device
		lr_up = lr_up.to(device)
		if lr_up.shape[0] == 1 and N > 1:
			lr_up = lr_up.expand(N, -1, -1, -1)
		elif lr_up.shape[0] > N:
			lr_up = lr_up[:N]
		x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=device)
		for step in range(self.n_times - 1, -1, -1):
			# progress = (self.n_times - 1 - step) / max(self.n_times - 1, 1)
			# print("█" * int(progress * 50) + "-" * int((1 - progress) * 50), end=f" {progress * 100:2.0f}%\r", flush=True)
			timestep = torch.full((N,), step, device=device, dtype=torch.long)
			x_t = self.denoise_at_t(x_t, timestep, lr_up)
		x_0 = self.reverse_scale_to_zero_to_one(x_t)
		return x_0


def _conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, t):
		# Map scalar timesteps into a smooth sinusoidal embedding (as in Transformer pos encodings).
		half = self.dim // 2
		freqs = torch.exp(
			torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / (half - 1))
		)
		emb = t[:, None].float() * freqs[None, :]
		return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.norm1 = nn.GroupNorm(32, in_channels)
		self.act1 = nn.SiLU()
		self.conv1 = _conv3x3(in_channels, out_channels)
		self.time_proj = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_dim, out_channels),
		)
		self.norm2 = nn.GroupNorm(32, out_channels)
		self.act2 = nn.SiLU()
		self.dropout = nn.Dropout(dropout)
		self.conv2 = _conv3x3(out_channels, out_channels)
		self.skip = (
			nn.Conv2d(in_channels, out_channels, kernel_size=1)
			if in_channels != out_channels
			else nn.Identity()
		)

	def forward(self, x, t_emb):
		# Residual path conditioned on the timestep embedding.
		h = self.conv1(self.act1(self.norm1(x)))
		t = self.time_proj(t_emb)[:, :, None, None]
		h = h + t
		h = self.conv2(self.dropout(self.act2(self.norm2(h))))
		return h + self.skip(x)


class AttentionBlock(nn.Module):
	def __init__(self, channels, num_heads=4):
		super().__init__()
		self.channels = channels
		self.num_heads = num_heads
		self.norm = nn.GroupNorm(32, channels)
		self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
		self.proj = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, x):
		b, c, h, w = x.shape
		h_in = self.norm(x)
		qkv = self.qkv(h_in)
		q, k, v = torch.chunk(qkv, 3, dim=1)
		# Flatten space and split heads for multi-head self-attention.
		q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
		k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
		v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
		scale = (c // self.num_heads) ** -0.5
		attn = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)
		h_out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
		h_out = h_out.reshape(b, c, h, w)
		h_out = self.proj(h_out)
		return x + h_out


class Downsample(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		return self.op(x)


class Upsample(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.op = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
		)

	def forward(self, x):
		return self.op(x)


class SR3UNet(nn.Module):
	def __init__(
		self,
		in_channels=3,
		cond_channels=3,
		base_channels=128,
		dim_mults=(1, 2, 2, 4),
		res_blocks_per_scale=2,
		attention_scales=(16,),
		time_emb_dim=512,
		dropout=0.0,
	):
		super().__init__()
		self.time_mlp = nn.Sequential(
			# Two-layer MLP over the sinusoidal embedding gives the network more expressive time features.
			SinusoidalPosEmb(time_emb_dim),
			nn.Linear(time_emb_dim, time_emb_dim),
			nn.SiLU(),
			nn.Linear(time_emb_dim, time_emb_dim),
		)

		self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)
		self.down_blocks = nn.ModuleList()
		self.up_blocks = nn.ModuleList()
		self.attention_scales = set(attention_scales)
		self.scale_resolutions = []

		channels = [base_channels]
		for mult in dim_mults:
			channels.append(base_channels * mult)

		self.downs = nn.ModuleList()
		self.skips = []

		in_ch = base_channels
		for idx, mult in enumerate(dim_mults):
			out_ch = base_channels * mult
			res_blocks = nn.ModuleList()
			for _ in range(res_blocks_per_scale):
				res_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
				in_ch = out_ch
				if (2 ** idx) in self.attention_scales:
					res_blocks.append(AttentionBlock(in_ch))
			down = nn.ModuleDict(
				{
					"res": res_blocks,
					"down": Downsample(in_ch) if idx < len(dim_mults) - 1 else nn.Identity(),
				}
			)
			self.downs.append(down)
			self.scale_resolutions.append(2 ** idx)

		self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)
		self.mid_attn = AttentionBlock(in_ch)
		self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)

		for idx, mult in reversed(list(enumerate(dim_mults))):
			out_ch = base_channels * mult
			res_blocks = nn.ModuleList()
			for _ in range(res_blocks_per_scale):
				res_blocks.append(ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim, dropout))
				in_ch = out_ch
				if (2 ** idx) in self.attention_scales:
					res_blocks.append(AttentionBlock(in_ch))
			up = nn.ModuleDict(
				{
					"res": res_blocks,
					"up": Upsample(in_ch) if idx > 0 else nn.Identity(),
				}
			)
			self.up_blocks.append(up)

		self.final_norm = nn.GroupNorm(32, base_channels)
		self.final_act = nn.SiLU()
		self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

	def forward(self, x, timesteps, condition):
		if condition is not None:
			x = torch.cat((x, condition), dim=1)
		t_emb = self.time_mlp(timesteps.float())
		h = self.input_conv(x)
		skips = []
		# Encoder: push each residual output onto the skip stack, keeping matching feature maps for the decoder.
		for block in self.downs:
			for layer in block["res"]:
				if isinstance(layer, ResidualBlock):
					h = layer(h, t_emb)
					skips.append(h)
				else:
					h = layer(h)
			h = block["down"](h)

		h = self.mid_block1(h, t_emb)
		h = self.mid_attn(h)
		h = self.mid_block2(h, t_emb)

		for block in self.up_blocks:
			for layer in block["res"]:
				if isinstance(layer, ResidualBlock):
					skip = skips.pop()
					h = torch.cat((h, skip), dim=1)
					h = layer(h, t_emb)
				else:
					h = layer(h)
			h = block["up"](h)

		h = self.final_conv(self.final_act(self.final_norm(h)))
		return h