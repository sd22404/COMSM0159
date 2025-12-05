import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM, LearnedPerceptualImagePatchSimilarity as LPIPS

from multiprocessing import cpu_count

from models.dip import UNet
from utils.datasets import DIPDataset
from utils.trainers import DIPTrainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float, help="Learning rate")
	parser.add_argument("--epochs", "-e", default=2000, type=int, help="Number of training epochs")
	parser.add_argument("--amp", default=True, help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", default=True, help="Use gradient scaler for mixed precision training")
	parser.add_argument("--checkpoint-prefix", "-n", default="dip", type=str, help="Prefix for checkpoint filename")
	parser.add_argument("--noise-std", default=0.0, type=float, help="Standard deviation of noise added to LR images during training")
	parser.add_argument("--downscale", default=8, type=int, help="Downscaling factor between HR and LR images")
	parser.add_argument("--load-checkpoint", "-c", default=None, type=Path, help="Load from checkpoint if available")
	parser.add_argument("--image-number", "-i", default=1, type=int, help="Number of the image to use from the dataset (1-indexed)")
	parser.add_argument("--val-interval", "-v", default=100, type=int, help="Validation interval in epochs")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# metrics
	psnr = PSNR(data_range=1.0).to(device)
	ssim = SSIM(data_range=1.0).to(device)
	lpips = LPIPS().to(device)

	val_dataset = DIPDataset(
		hr_root="dataset/DIV2K_valid_HR",
		lr_root="dataset/DIV2K_valid_LR_x8",
		noise_std=args.noise_std,
		downscale=args.downscale,
		idx=args.image_number - 1
	)

	# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=cpu_count())
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cpu_count())

	_, hr = next(iter(val_loader))
	H, W = hr.shape[-2:]
	z_channels = 32
	z_sigma = 0.05
	base_channels = 128
	skip_channels = 4
	model = UNet(height=H, width=W, base_channels=base_channels, z_channels=z_channels, skip_channels=skip_channels, z_sigma=z_sigma).to(device)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	start = 0
	if args.load_checkpoint is not None and args.load_checkpoint.exists():
		checkpoint = torch.load(args.load_checkpoint, map_location=device)
		start = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print(f"Loaded checkpoint from {args.load_checkpoint}")

	trainer = DIPTrainer(
		model=model,
		criterion=criterion,
		optimizer=optimizer,
		train_loader=val_loader,
		val_loader=val_loader,
		scheduler=None,
		device=device,
		psnr=psnr,
		ssim=ssim,
		lpips=lpips,
		use_amp=args.amp,
		use_grad_scaler=args.grad_scaler,
		val_interval=args.val_interval,
		save_interval=args.epochs + 1, # don't save model
		checkpoint_prefix=args.checkpoint_prefix,
		img=args.image_number,
	)
	
	trainer.train(start, args.epochs)

if __name__ == "__main__":
	main(parse_args())