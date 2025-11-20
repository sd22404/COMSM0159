from time import time
import os
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class DIPTrainer:
	def __init__(self, model, criterion, optimizer, device):
		self.device = device
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.best_loss = float('inf')

	def train(self, lr, hr, epochs):
		self.noise = torch.randn((1, hr.shape[1], hr.shape[2], hr.shape[3])).to(self.device).detach()
		self.model.train()
		stagnant = 0
		patience = 50
		for epoch in range(1, epochs + 1):
			t0 = time()
			self.optimizer.zero_grad()

			hr_out = self.model(self.noise)
			lr_out = F.resize(hr_out, size=lr.shape[2:])

			loss = self.criterion(lr_out, lr.to(self.device))
			loss.backward()
			self.optimizer.step()

			tT = time() - t0
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.6f}, Time: {tT:.2f} seconds")

			if loss.item() < self.best_loss:
				self.best_loss = loss.item()
				stagnant = 0
			else:
				stagnant += 1

			if stagnant >= patience:
				print(f"Early stopping: no improvement in {patience} epochs (stopping at epoch {epoch}).")
				break
	
	def visualise(self, lr, hr):
		self.model.eval()
		with torch.no_grad():
			output = self.model(self.noise)
			output = output.cpu().clamp(0, 1)

		fig, axes = plt.subplots(1, 3, figsize=(12,4))

		axes[0].imshow(output.squeeze().permute(1,2,0))
		# axes[0].set_title("DIP Output")
		
		axes[1].imshow(lr.squeeze().permute(1,2,0))
		# axes[1].set_title("Low Resolution")
		
		axes[2].imshow(hr.squeeze().permute(1,2,0))
		# axes[2].set_title("High Resolution")
		
		os.makedirs("results", exist_ok=True)
		plt.savefig("results/dip_out.png")