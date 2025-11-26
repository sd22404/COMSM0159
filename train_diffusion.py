import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
import lpips as lp

from multiprocessing import cpu_count

from models.old.diffusion import Denoiser, Diffusion
from models.diffusion import SR3UNet, Diffuser
from utils.dataset import DIV2K_X8
from utils.trainers import DiffusionTrainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=16, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=10, type=int, help="Number of training epochs")
	parser.add_argument("--iterations", "-i", default=2000, type=int, help="Number of diffusion steps")
	parser.add_argument("--amp", default=True, help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", default=True, help="Use gradient scaler for mixed precision training")
	parser.add_argument("--schedule", action='store_true', help="Use learning rate scheduler")
	parser.add_argument("--no-train", action='store_true', help="Skip training and only run visualization")
	parser.add_argument("--load-checkpoint", "-c", default=None, type=Path, help="Load from checkpoint if available")
	parser.add_argument("--checkpoint-dir", default="checkpoints", type=Path, help="Path to save checkpoint")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	img_size = 256
	hr_crop = T.CenterCrop(img_size)
	lr_crop = T.CenterCrop(img_size // 8)

	# metrics
	psnr = PSNR(data_range=1.0).to(device)
	ssim = SSIM(data_range=1.0).to(device)
	lpips = lp.LPIPS().to(device)

	train_dataset = DIV2K_X8(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
		hr_transform=hr_crop,
		lr_transform=lr_crop
	)
	val_dataset = DIV2K_X8(
		hr_root="dataset/DIV2K_valid_HR",
		lr_root="dataset/DIV2K_valid_LR_x8",
		hr_transform=hr_crop,
		lr_transform=lr_crop
	)
	
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count())
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count())
	
	# model = Denoiser(image_resolution=(img_size, img_size, 3))
	# diff = Diffusion(denoiser=model, image_resolution=(img_size, img_size, 3), n_times=args.iterations).to(device)
	model = SR3UNet(base_channels=64, res_blocks_per_scale=1)
	diff = Diffuser(model, image_resolution=(img_size, img_size, 3), n_times=args.iterations).to(device)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.schedule else None

	start = 0
	if args.load_checkpoint is not None and args.load_checkpoint.exists():
		checkpoint = torch.load(args.load_checkpoint, map_location=device)
		# start = checkpoint['epoch'] + 1
		diff.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print(f"Loaded checkpoint from {args.load_checkpoint}")

	trainer = DiffusionTrainer(
		diff=diff,
		criterion=criterion,
		optimizer=optimizer,
		scheduler=scheduler,
		train_loader=train_loader,
		val_loader=val_loader,
		device=device,
		psnr=psnr,
		ssim=ssim,
		lpips=lpips,
		use_amp=args.amp,
		use_grad_scaler=args.grad_scaler,
		val_interval=args.epochs,
		save_interval=max(args.epochs // 10, 1),
		checkpoint_prefix="diffusion",
	)

	if not args.no_train:
		trainer.train(start, args.epochs)

if __name__ == "__main__":
	main(parse_args())