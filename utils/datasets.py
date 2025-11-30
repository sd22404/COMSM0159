import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import utils.utils as utils

class PairDataset(Dataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), crop_size=None, noise_std=0.0, downscale=8.0):
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

		if self.crop_size is not None:
			i, j, h, w = T.RandomCrop.get_params(hr, output_size=(self.crop_size, self.crop_size))
			hr = T.functional.crop(hr, i, j, h, w)
			i //= self.downscale; j //= self.downscale; h //= self.downscale; w //= self.downscale
			lr = T.functional.crop(lr, i, j, h, w)

		hr = T.Compose([self.hr_transform, T.ToTensor()])(hr)
		lr = T.Compose([self.lr_transform, T.ToTensor()])(lr)

		if self.downscale != 8.0:
			lr = T.functional.resize(
				lr,
				[int(lr.size(1) // (self.downscale / 8.0)), int(lr.size(2) // (self.downscale / 8.0))],
				interpolation=T.InterpolationMode.BICUBIC
			)

		if self.noise_std > 0:
			noise = torch.randn_like(lr) * self.noise_std
			lr = lr + noise
			lr = torch.clamp(lr, 0, 1)

		return lr, hr


class DIPDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), crop_size=None, noise_std=0.0, downscale=8.0, idx=0):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, crop_size, noise_std, downscale)
		self.idx = idx
	
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		return super().__getitem__(self.idx)


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