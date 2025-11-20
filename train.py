import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from multiprocessing import cpu_count

from models.dip import DIP
from utils.dataset import DIV2K_X8
from utils.trainers import DIPTrainer

IMAGE_SIZE = 224

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float, help="Learning rate")
	parser.add_argument("--batch-size", "-bs", default=16, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--epochs", "-e", default=20, type=int, help="Number of training epochs")
	return parser.parse_args()

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	transforms = T.Compose([
		T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		T.ToTensor(),
	])

	train_dataset = DIV2K_X8(
		root_hr="dataset/DIV2K_train_HR",
		root_lr="dataset/DIV2K_train_LR_x8",
		transform=transforms,
	)
	dip_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=cpu_count())
	

	model = DIP(height=IMAGE_SIZE, width=IMAGE_SIZE, channels=3).to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

	trainer = DIPTrainer(model, criterion, optimizer, dip_loader, device)
	trainer.train(epochs=args.epochs)
	trainer.visualise()

if __name__ == "__main__":
	main(parse_args())