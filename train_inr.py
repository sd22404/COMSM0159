import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM, LearnedPerceptualImagePatchSimilarity as LPIPS

from multiprocessing import cpu_count

from models.inr import LIIF
from utils.datasets import INRDataset
from utils.trainers import INRTrainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-4, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=4, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=1000, type=int, help="Number of training epochs")
	parser.add_argument("--load-checkpoint", "-c", default=None, type=Path, help="Load from checkpoint if available")
	parser.add_argument("--checkpoint-dir", default="checkpoints", type=Path, help="Path to save checkpoint")
	parser.add_argument("--amp", default=True, help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", default=True, help="Use gradient scaler for mixed precision training")
	parser.add_argument("--schedule", default=True, type=bool, help="Use learning rate scheduler")
	parser.add_argument("--no-train", action='store_true', help="Skip training and only run visualization")
	parser.add_argument("--sample-size", default=576, type=int, help="Number of pixels to sample per image during training")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	img_size = 384
	hr_crop = T.CenterCrop(img_size)
	lr_crop = T.CenterCrop(img_size // 8)

	# metrics
	psnr = PSNR(data_range=1.0).to(device)
	ssim = SSIM(data_range=1.0).to(device)
	lpips = LPIPS().to(device)

	train_dataset = INRDataset(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
		hr_transform=hr_crop,
		lr_transform=lr_crop,
	)
	val_dataset = INRDataset(
		hr_root="dataset/DIV2K_valid_HR",
		lr_root="dataset/DIV2K_valid_LR_x8",
		hr_transform=hr_crop,
		lr_transform=lr_crop,
	)
	
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count())
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count())
	
	model = LIIF().to(device)
	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5) if args.schedule else None

	start = 0
	if args.load_checkpoint is not None and args.load_checkpoint.exists():
		checkpoint = torch.load(args.load_checkpoint, map_location=device)
		start = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print(f"Loaded checkpoint from {args.load_checkpoint}")

	trainer = INRTrainer(
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
		save_interval=args.epochs // 10,
		checkpoint_prefix="inr",
		sample_size=args.sample_size
	)

	if not args.no_train:
		trainer.train(start, args.epochs)

	trainer.val()

if __name__ == "__main__":
	main(parse_args())