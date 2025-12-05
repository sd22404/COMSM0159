import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import utils.utils as utils

class PairDataset(Dataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), crop_size=None, noise_std=0.0, downscale=8):
		self.hr_root = hr_root
		self.lr_root = lr_root
		self.hr_files = sorted(os.listdir(hr_root))
		self.lr_files = sorted(os.listdir(lr_root))
		self.crop_size = crop_size
		self.noise_std = noise_std
		self.downscale = downscale

		assert len(self.hr_files) == len(self.lr_files), \
			"HR and LR folders must contain equal number of images."

		self.hr_transform = hr_transform
		self.lr_transform = lr_transform

	def __len__(self):
		return len(self.hr_files)

	def __getitem__(self, idx):
		hr_path = os.path.join(self.hr_root, self.hr_files[idx])
		lr_path = os.path.join(self.lr_root, self.lr_files[idx])

		hr = Image.open(hr_path).convert("RGB")
		lr = Image.open(lr_path).convert("RGB")

		if self.downscale < 8:
			h = max(1, hr.height // self.downscale)
			w = max(1, hr.width // self.downscale)
			lr = T.functional.resize(hr, (h, w), interpolation=T.InterpolationMode.BICUBIC)
		elif self.downscale > 8:
			h = max(1, lr.height * 8 // self.downscale)
			w = max(1, lr.width * 8 // self.downscale)
			lr = T.functional.resize(lr, (h, w), interpolation=T.InterpolationMode.BICUBIC)

		if self.crop_size is not None:
			# aligned random crop
			i, j, _, _ = T.RandomCrop.get_params(hr, output_size=(self.crop_size, self.crop_size))
			i = (i // self.downscale) * self.downscale
			j = (j // self.downscale) * self.downscale
			i = min(i, hr.height - self.crop_size)
			j = min(j, hr.width - self.crop_size)
			hr = T.functional.crop(hr, i, j, self.crop_size, self.crop_size)
			i //= self.downscale
			j //= self.downscale
			h = self.crop_size // self.downscale
			w = self.crop_size // self.downscale
			lr = T.functional.crop(lr, i, j, h, w)

		hr = T.Compose([self.hr_transform, T.ToTensor()])(hr)
		lr = T.Compose([self.lr_transform, T.ToTensor()])(lr)

		if self.noise_std > 0:
			noise = torch.randn_like(lr) * self.noise_std
			lr = lr + noise
			lr = torch.clamp(lr, 0, 1)

		return lr, hr


class DIPDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), crop_size=None, noise_std=0.0, downscale=8.0, idx=0):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, crop_size, 0.0, downscale)
		self.idx = idx
		self.fixed_noise = None
		self.fixed_noise_std = noise_std
		# use the same noise at every stage for DIP
		if self.fixed_noise_std > 0:
			lr, _ = super().__getitem__(self.idx)
			self.fixed_noise = torch.randn_like(lr) * self.fixed_noise_std
	
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		lr, hr = super().__getitem__(self.idx)

		if self.fixed_noise is not None:
			lr = lr + self.fixed_noise
			lr = torch.clamp(lr, 0, 1)

		return lr, hr


class INRDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), crop_size=None, noise_std=0.0, downscale=8.0, sample_size=None):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, crop_size, noise_std, downscale)
		self.sample_size = sample_size
	
	def __getitem__(self, idx):
		lr, hr = super().__getitem__(idx)
		H, W = hr.shape[1], hr.shape[2]
		coords = utils.make_grid(H, W).view(-1, 2)
		cells = utils.make_cells(coords, H, W)
		hr_pix = hr.view(3, -1).permute(1, 0)

		if self.sample_size is not None and len(coords) > self.sample_size:
			indices = torch.randperm(len(coords))[:self.sample_size]
			cells = cells[indices]
			coords = coords[indices]
			hr_pix = hr_pix[indices]

		return lr, hr_pix, coords, cells