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
	def __init__(self, root_hr, root_lr, transform=None):
		self.root_hr = root_hr
		self.root_lr = root_lr
		self.hr_files = sorted(os.listdir(root_hr))
		self.lr_files = sorted(os.listdir(root_lr))

		assert len(self.hr_files) == len(self.lr_files), \
			"HR and LR folders must contain equal number of images."

		self.transform = transform or transforms.ToTensor()

	def __len__(self):
		return len(self.hr_files)

	def __getitem__(self, idx):
		hr_path = os.path.join(self.root_hr, self.hr_files[idx])
		lr_path = os.path.join(self.root_lr, self.lr_files[idx])

		hr = Image.open(hr_path).convert("RGB")
		lr = Image.open(lr_path).convert("RGB")

		# hr = self.transform(hr)
		hr = transforms.ToTensor()(hr)
		hr = transforms.CenterCrop(min(hr.shape[1], hr.shape[2]))(hr)
		lr = self.transform(lr)

		return lr, hr