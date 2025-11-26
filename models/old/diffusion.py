import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
	def __init__(self, denoiser, image_resolution, n_times, beta_minmax=[1e-4, 2e-2], device='cuda'):
		super(Diffusion, self).__init__()
		self.n_times = n_times
		self.img_H, self.img_W, self.img_C = image_resolution
		self.model = denoiser
		# Define linear variance schedule (betas)
		beta_1, beta_T = beta_minmax
		betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device) # follows DDPM paper: cosine function instead of linear?
		self.sqrt_betas = torch.sqrt(betas)
		# Define alphas for forward diffusion process
		self.alphas = 1 - betas
		self.sqrt_alphas = torch.sqrt(self.alphas)
		alpha_bars = torch.cumprod(self.alphas, dim=0)
		self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
		self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
		self.device = device

	def extract(self, a, t, x_shape):
		# Extract the specific values for the batch of time-steps `t`
		b, *_ = t.shape
		out = a.gather(-1, t)
		return out.reshape(b, *((1,) * (len(x_shape) - 1)))

	def scale_to_minus_one_to_one(self, x):
		# Scale input `x` from [0, 1] to [-1, 1]
		return x * 2 - 1

	def reverse_scale_to_zero_to_one(self, x):
		# Scale input `x` from [-1, 1] back to [0, 1]
		return (x + 1) * 0.5

	def make_noisy(self, x_zeros, t):
		# Perturb `x_0` into `x_t` (forward diffusion process)
		epsilon = torch.randn_like(x_zeros).to(self.device)
		sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
		sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
		# Let's make noisy sample!: i.e., Forward process with fixed variance schedule
		#      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
		noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
		return noisy_sample.detach(), epsilon


	def forward(self, x_zeros, cond):
		x_zeros = self.scale_to_minus_one_to_one(x_zeros)
		B, _, _, _ = x_zeros.shape
		# 1. Randomly select a diffusion time-step `t`
		t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
		# 2. Forward diffusion: perturb `x_zeros` using the fixed variance schedule
		perturbed_images, epsilon = self.make_noisy(x_zeros, t)
		# 3. Predict the noise (`epsilon`) given the perturbed image at time-step `t`
		pred_epsilon = self.model(perturbed_images, t, cond)
		return perturbed_images, epsilon, pred_epsilon


	def denoise_at_t(self, x_t, timestep, t, cond):
		B, _, _, _ = x_t.shape
		if cond is not None:
			cond = self.scale_to_minus_one_to_one(cond)
			cond = cond.to(self.device)
			if cond.shape[0] != B:
				cond = cond.expand(B, -1, -1, -1)
		# Generate random noise `z` for sampling, except for the final step (`t=0`)
		if t > 1:
			z = torch.randn_like(x_t).to(self.device)
		else:
			z = torch.zeros_like(x_t).to(self.device)
		# at inference, we use predicted noise(epsilon) to restore perturbed data sample.
		# Use the model to predict noise (`epsilon_pred`) given `x_t` at `timestep`
		epsilon_pred = self.model(x_t, timestep, cond)
		alpha = self.extract(self.alphas, timestep, x_t.shape)
		sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
		sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
		sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
		# denoise at time t, denoise `x_t` to estimate `x_{t-1}`
		x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
		return x_t_minus_1.clamp(-1., 1)

	def sample(self, N, lr_up):
		lr_up = lr_up.to(self.device)
		if lr_up.shape[0] == 1 and N > 1:
			lr_up = lr_up.expand(N, -1, -1, -1)
		elif lr_up.shape[0] > N:
			lr_up = lr_up[:N]
		# Start from random noise vector `x_T`, x_0 (for simplicity, x_T declared as x_t instead of x_T)
		x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
		# Autoregressively denoise from `x_T` to `x_0`
		#     i.e., generate image from noise, x_T
		for t in range(self.n_times-1, -1, -1):
			progress = (self.n_times - 1 - t) / (self.n_times - 1)
			print("█" * int(progress * 50) + "-" * int((1 - progress) * 50), end=f' {progress * 100:2.0f}%\r', flush=True)
			timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
			x_t = self.denoise_at_t(x_t, timestep, t, lr_up)
		# Convert the final result `x_0` back to [0, 1] range
		x_0 = self.reverse_scale_to_zero_to_one(x_t)
		return x_0


class Denoiser(nn.Module):
	def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim = 256):
		super(Denoiser, self).__init__()
		_, _, img_C = image_resolution
		self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
		self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
		self.cond_project = ConvBlock(img_C, hidden_dims[0], kernel_size=3, activation_fn=True)
		self.merge = ConvBlock(in_channels=hidden_dims[0]*2, out_channels=hidden_dims[0], kernel_size=1)
		self.time_project = nn.Sequential(
								 ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
								 ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))
		self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])
		for idx in range(1, len(hidden_dims)):
			self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3, dilation=3**((idx-1)//2),
													activation_fn=True, gn=True, gn_groups=8))
		self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)

	def forward(self, perturbed_x, diffusion_timestep, condition):
		condition_feat = self.cond_project(condition)
		y = self.in_project(perturbed_x)
		merged = torch.cat((y, condition_feat), dim=1)
		y = self.merge(merged)
		diffusion_embedding = self.time_embedding(diffusion_timestep)
		diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
		for i in range(len(self.convs)):
			y = self.convs[i](y, diffusion_embedding, residual = True)
		y = self.out_project(y)
		return y

class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		device = x.device
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
		emb = x[:, None] * emb[None, :]
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb

class ConvBlock(nn.Conv2d):
	"""
		Conv2D Block
			Args:
				x: (N, C_in, H, W)
			Returns:
				y: (N, C_out, H, W)
	"""
	def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
					stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):

		if padding == 'same':
			padding = kernel_size // 2 * dilation

		super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
											stride=stride, padding=padding, dilation=dilation,
											groups=groups, bias=bias)

		self.activation_fn = nn.SiLU() if activation_fn else None
		self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

	def forward(self, x, time_embedding=None, residual=False):
		if residual:
			x = x + time_embedding
			y = x
			x = super(ConvBlock, self).forward(x)
			y = y + x
		else:
			y = super(ConvBlock, self).forward(x)
		y = self.group_norm(y) if self.group_norm is not None else y
		y = self.activation_fn(y) if self.activation_fn is not None else y
		return y