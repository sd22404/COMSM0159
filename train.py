import argparse
import torch
from torch.utils.data import DataLoader

from models.example_model import ExampleModel
from utils.dataset import DIV2K_X8
from utils.trainer import Trainer


def parse_args():
	parser = argparse.ArgumentParser()
	return parser.parse_args()


def main():
	args = parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_dataset = DIV2K_X8(
		root_hr="dataset/DIV2K_train_HR",
		root_lr="dataset/DIV2K_train_LR_x8",
	)
	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

	test_dataset = DIV2K_X8(
		root_hr="dataset/DIV2K_valid_HR",
		root_lr="dataset/DIV2K_valid_LR_x8",
	)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

	model = ExampleModel().to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

	trainer = Trainer(model, criterion, optimizer, train_loader, test_loader, device)
	trainer.train(num_epochs=100)

if __name__ == "__main__":
	main()