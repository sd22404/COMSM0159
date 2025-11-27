import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class PairDataset(Dataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), square=False):
		self.hr_root = hr_root
		self.lr_root = lr_root
		self.hr_files = sorted(os.listdir(hr_root))
		self.lr_files = sorted(os.listdir(lr_root))
		self.square = square

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

		if self.square:
			hr_size = min(hr.size[0], hr.size[1])
			lr_size = min(lr.size[0], lr.size[1])
			hr = T.CenterCrop(hr_size)(hr)
			lr = T.CenterCrop(lr_size)(lr)

		hr = self.hr_transform(hr)
		lr = self.lr_transform(lr)

		return lr, hr

class DIPDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), idx=0):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform)
		self.idx = idx
	
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		return super().__getitem__(self.idx)

class INRDataset(PairDataset):
	def __init__(self, hr_root, lr_root, hr_transform=T.Lambda(lambda x: x), lr_transform=T.Lambda(lambda x: x), square=False):
		super().__init__(hr_root, lr_root, hr_transform, lr_transform, square=square)
	
	def _make_grid(self, height, width):
		y = torch.linspace(0.0, 1.0, steps=height)
		x = torch.linspace(0.0, 1.0, steps=width)
		grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
		coords = torch.stack((grid_x, grid_y), dim=-1)
		coords = coords.view(height * width, 2)
		return coords
	
	def __getitem__(self, idx):
		lr, hr = super().__getitem__(idx)
		coords = self._make_grid(hr.shape[1], hr.shape[2])
		return lr, hr, coords