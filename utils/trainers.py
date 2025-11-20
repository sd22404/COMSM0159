from time import time
import os
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class DIPTrainer:
	def __init__(self, model, width, height, criterion, optimizer, device):
		self.device = device
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.noise = torch.randn(1, 3, width, height).to(device)

	def train(self, lr, epochs):
		self.model.train()
		for epoch in range(epochs):
			epoch_start_time = time()
			self.optimizer.zero_grad()

			output = self.model(self.noise)
			lr_output = F.resize(output, size=lr.shape[2:])
			loss = self.criterion(lr_output, lr.to(self.device))
			loss.backward()
			self.optimizer.step()

			epoch_train_time = time() - epoch_start_time
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.6f}, Time: {epoch_train_time:.2f} seconds")
	
	def visualise(self, lr, hr):
		self.model.eval()
		with torch.no_grad():
			output = self.model(self.noise)
		output = output.cpu()
		fig = plt.figure()
		ax1 = fig.add_subplot(2,2,1)
		ax2 = fig.add_subplot(2,2,2)
		ax3 = fig.add_subplot(2,2,3)
		ax1.imshow(output.squeeze().permute(1,2,0))
		ax1.set_title("DIP Output")
		ax2.imshow(lr.squeeze().permute(1,2,0))
		ax2.set_title("Low Resolution")
		ax3.imshow(hr.squeeze().permute(1,2,0))
		ax3.set_title("High Resolution")
		os.makedirs("results", exist_ok=True)
		plt.savefig("results/dip_out.png")