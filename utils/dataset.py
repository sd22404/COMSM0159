import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DIV2K_X8(Dataset):
	"""
	Loads paired LR (x8) and HR images from DIV2K 2018 dataset.
	Folder structure:
		root_hr/0001.png
		root_lr/0001x8.png
	"""
	def __init__(self, hr_root, lr_root, hr_transform=transforms.Lambda(lambda x: x), lr_transform=transforms.Lambda(lambda x: x)):
		self.hr_root = hr_root
		self.lr_root = lr_root
		self.hr_files = sorted(os.listdir(hr_root))
		self.lr_files = sorted(os.listdir(lr_root))

		assert len(self.hr_files) == len(self.lr_files), \
			"HR and LR folders must contain equal number of images."

		self.hr_transform = transforms.Compose([
			hr_transform, 
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		self.lr_transform = transforms.Compose([
			lr_transform, 
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def __len__(self):
		return len(self.hr_files)

	def __getitem__(self, idx):
		hr_path = os.path.join(self.hr_root, self.hr_files[idx])
		lr_path = os.path.join(self.lr_root, self.lr_files[idx])

		hr = Image.open(hr_path).convert("RGB")
		lr = Image.open(lr_path).convert("RGB")

		hr = self.hr_transform(hr)
		lr = self.lr_transform(lr)

		return lr, hr