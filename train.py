import argparse
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from models.example_model import ExampleModel
from utils.dataset import DIV2K_X8
from utils.trainers import Trainer


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
	parser.add_argument("--batch-size", default=16, type=int, help="Number of images within each mini-batch")
	parser.add_argument("--num-epochs", default=20, type=int, help="Number of training epochs")
	return parser.parse_args()


def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_dataset = DIV2K_X8(
		root_hr="dataset/DIV2K_train_HR",
		root_lr="dataset/DIV2K_train_LR_x8",
	)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count())

	val_dataset = DIV2K_X8(
		root_hr="dataset/DIV2K_valid_HR",
		root_lr="dataset/DIV2K_valid_LR_x8",
	)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cpu_count())

	model = ExampleModel().to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

	trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device)
	trainer.train(num_epochs=args.num_epochs)

if __name__ == "__main__":
	main(parse_args())