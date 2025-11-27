import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
import lpips as lp

from multiprocessing import cpu_count

from models.dip import UNet, skip
from utils.dataset import PairDataset, DIPDataset
from utils.trainers import DIPTrainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=1, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=4000, type=int, help="Number of training epochs")
	parser.add_argument("--scheduler", action='store_true', help="Use learning rate scheduler")
	parser.add_argument("--amp", default=True, help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", default=True, help="Use gradient scaler for mixed precision training")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	img_size = 256
	hr_crop = T.CenterCrop(img_size)
	lr_crop = T.CenterCrop(img_size // 8)

	# metrics
	psnr = PSNR(data_range=1.0).to(device)
	ssim = SSIM(data_range=1.0).to(device)
	lpips = lp.LPIPS().to(device)

	train_dataset = DIPDataset(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
		# hr_transform=hr_crop,
		# lr_transform=lr_crop,
		idx=1
	)
	val_dataset = DIPDataset(
		hr_root="dataset/DIV2K_valid_HR",
		lr_root="dataset/DIV2K_valid_LR_x8",
		# hr_transform=hr_crop,
		# lr_transform=lr_crop,
		idx=1
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count())
	val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count())

	lr, hr = next(iter(val_loader))
	img = hr.squeeze(0)
	z_channels = 32
	z_sigma = 0.05
	# base_channels = 24	
	# model = UNet(height=hr.shape[2], width=hr.shape[3], base_channels=base_channels, z_channels=z_channels, z_sigma=z_sigma).to(device)
	
	model = skip(num_input_channels=z_channels).to(device)
	model.register_buffer('z', torch.randn((1, z_channels, img.shape[1], img.shape[2])).to(device) * 0.1)
	model.get_z = lambda sigma=0.0: model.z + sigma * torch.randn_like(model.z)
	_original_forward = model.forward
	def _forward_with_noise(z=None):
		z = model.get_z(z_sigma) if z is None else z
		return _original_forward(z)
	model.forward = _forward_with_noise

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 100, gamma=0.8) if args.scheduler else None

	trainer = DIPTrainer(
		model=model,
		criterion=criterion,
		optimizer=optimizer,
		train_loader=train_loader,
		val_loader=val_loader,
		scheduler=scheduler,
		device=device,
		psnr=psnr,
		ssim=ssim,
		lpips=lpips,
		use_amp=args.amp,
		use_grad_scaler=args.grad_scaler,
		val_interval=args.epochs // 10,
		save_interval=args.epochs // 100,
		checkpoint_prefix="dip"
	)
	trainer.train(0, args.epochs)

if __name__ == "__main__":
	main(parse_args())