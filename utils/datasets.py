import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class PairDataset(Dataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), noise_std=0.0, downscale=1.0):
		self.hr_root = hr_root
		self.lr_root = lr_root
		self.hr_files = sorted(os.listdir(hr_root))
		self.lr_files = sorted(os.listdir(lr_root))
		self.noise_std = noise_std
		self.downscale = downscale

		assert len(self.hr_files) == len(self.lr_files), \
			"HR and LR folders must contain equal number of images."

		self.hr_transform = T.Compose([
			hr_transform,
			T.ToTensor(),
			# T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		self.lr_transform = T.Compose([
			lr_transform,
			T.ToTensor(),
			# T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def __len__(self):
		return len(self.hr_files)

	def __getitem__(self, idx):
		hr_path = os.path.join(self.hr_root, self.hr_files[idx])
		lr_path = os.path.join(self.lr_root, self.lr_files[idx])

		hr = Image.open(hr_path).convert("RGB")
		lr = Image.open(lr_path).convert("RGB")
		
		if self.downscale > 1:
			lr = T.functional.resize(
				lr, 
				[lr.shape[1] // self.downscale, lr.shape[2] // self.downscale], 
				interpolation=T.InterpolationMode.BICUBIC
			)

		if self.noise_std > 0:
			noise = torch.randn_like(lr) * self.noise_std
			lr = lr + noise
			lr = torch.clamp(lr, 0, 1)

		hr = self.hr_transform(hr)
		lr = self.lr_transform(lr)

		return lr, hr


class DIPDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), idx=0, noise_std=0.0, downscale=1.0):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, noise_std, downscale)
		self.idx = idx
	
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		return super().__getitem__(self.idx)


class INRDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), sample_size=None, noise_std=0.0, downscale=1.0):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, noise_std, downscale)
		self.sample_size = sample_size
	
	def _make_grid(self, height, width):
		y = torch.linspace(-1.0, 1.0, steps=height)
		x = torch.linspace(-1.0, 1.0, steps=width)
		grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
		coords = torch.stack((grid_x, grid_y), dim=-1)
		coords = coords.view(height * width, 2)

		if self.sample_size is not None and len(coords) > self.sample_size:
			indices = torch.randperm(len(coords))[:self.sample_size]
			coords = coords[indices]

		return coords

	def _make_cells(self, coords, height, width):
		cells = torch.ones_like(coords)
		cells[:, 0] *= 2 / height
		cells[:, 1] *= 2 / width
		
		if self.sample_size is not None and len(cells) > self.sample_size:
			indices = torch.randperm(len(cells))[:self.sample_size]
			cells = cells[indices]
		
		return cells
	
	def __getitem__(self, idx):
		lr, hr = super().__getitem__(idx)
		coords = self._make_grid(hr.shape[1], hr.shape[2])
		cells = self._make_cells(coords, hr.shape[1], hr.shape[2])
		return lr, hr, coords, cells