import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from multiprocessing import cpu_count

from models.dip import UNet
from utils.dataset import DIV2K_X8
from utils.dip_trainer import DIPTrainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=1, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=2000, type=int, help="Number of training epochs")
	parser.add_argument("--scheduler", action='store_true', help="Use learning rate scheduler")
	parser.add_argument("--amp", action='store_true', help="Use automatic mixed precision")
	parser.add_argument("--grad-scaler", action='store_true', help="Use gradient scaler for mixed precision training")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_dataset = DIV2K_X8(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
	)
	
	dip_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=cpu_count())
	lr, hr = next(iter(dip_loader))

	model = UNet(height=hr.shape[2], width=hr.shape[3]).to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler else None

	trainer = DIPTrainer(model, criterion, optimizer, scheduler, device, use_amp=args.amp, use_grad_scaler=args.grad_scaler)
	trainer.train(lr, hr, args.epochs)

if __name__ == "__main__":
	main(parse_args())