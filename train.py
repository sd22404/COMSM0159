import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from multiprocessing import cpu_count

from models.dip import DIP
from utils.dataset import DIV2K_X8
from utils.trainers import DIPTrainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=1, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=1000, type=int, help="Number of training epochs")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_dataset = DIV2K_X8(
		hr_root="dataset/DIV2K_train_HR",
		lr_root="dataset/DIV2K_train_LR_x8",
		hr_transform=T.CenterCrop(1024),
	)
	
	dip_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=cpu_count())
	lr, hr = next(iter(dip_loader))

	model = DIP(height=hr.shape[2], width=hr.shape[3], channels=hr.shape[1]).to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	trainer = DIPTrainer(model, criterion, optimizer, device)
	trainer.train(lr, args.epochs)
	trainer.visualise(lr, hr)

if __name__ == "__main__":
	main(parse_args())