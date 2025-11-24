import argparse
import torch
import math
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
import lpips as lp

from multiprocessing import cpu_count

from models.unet import UNet
from models.skip import skip
from utils.dataset import DIV2K_X8
from utils.dip_trainer import DIPTrainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=1, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=4000, type=int, help="Number of training epochs")
	parser.add_argument("--scheduler", action='store_true', help="Use learning rate scheduler")
	parser.add_argument("--amp", action='store_true', help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", action='store_true', help="Use gradient scaler for mixed precision training")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# metrics
	psnr = PSNR(data_range=1.0).to(device)
	ssim = SSIM(data_range=1.0).to(device)
	lpips = lp.LPIPS().to(device)

	train_dataset = DIV2K_X8(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
	)
	
	dip_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=cpu_count())
	# lr, hr = next(iter(dip_loader))
	lr, hr = dip_loader.dataset[1]
	lr = lr.unsqueeze(0).to(device)
	hr = hr.unsqueeze(0).to(device)

	base_channels = 24
	z_channels = 32
	z_sigma = 0.05
	
	# model = UNet(height=hr.shape[2], width=hr.shape[3], base_channels=base_channels, z_channels=z_channels, z_sigma=z_sigma).to(device)
	
	model = skip(num_input_channels=z_channels).to(device)
	model.register_buffer('z', torch.randn((1, z_channels, hr.shape[2], hr.shape[3])).to(device) * 0.1)
	model.get_z = lambda sigma=0.0: model.z + sigma * torch.randn_like(model.z)
	_original_forward = model.forward
	def _forward_with_noise(z=None):
		z = model.get_z(z_sigma) if z is None else z
		return _original_forward(z)
	model.forward = _forward_with_noise

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler else None

	trainer = DIPTrainer(model, criterion, optimizer, scheduler, device, psnr, ssim, lpips, use_amp=args.amp, use_grad_scaler=args.grad_scaler)
	trainer.train(lr, hr, args.epochs)

if __name__ == "__main__":
	main(parse_args())