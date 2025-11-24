from time import time
import os
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class DIPTrainer:
	def __init__(self, model, criterion, optimizer, scheduler, device, use_amp=False, use_grad_scaler=False):
		self.device = device
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.best_loss = float('inf')
		self.best_state = None
		self.scaler = torch.amp.GradScaler(enabled=(device == 'cuda' and use_grad_scaler))
		self.amp = use_amp
		self.scheduler = scheduler

	def train(self, lr, hr, epochs):
		self.model.train()
		stagnant = 0
		patience = epochs // 10
		for epoch in range(epochs):
			t0 = time()
			self.optimizer.zero_grad()

			with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == 'cuda' and self.amp)):
				hr_out = self.model(self.model.z)
				lr_out = F.resize(hr_out, size=lr.shape[2:], interpolation=F.InterpolationMode.BICUBIC)
				loss = self.criterion(lr_out, lr.to(self.device))
		
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()

			tT = time() - t0
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.6f}, Time: {tT:.2f} seconds")
			if (epoch % 100 == 0): self.visualise(lr, hr, hr_out.cpu().detach().clamp(0, 1))

			if loss.item() < self.best_loss:
				self.best_loss = loss.item()
				self.best_state = {
					'model': self.model.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'epoch': epoch,
					'loss': self.best_loss
				}

				stagnant = 0
			else:
				stagnant += 1

			if stagnant >= patience:
				print(f"Early stopping: no improvement in {patience} epochs (stopping at epoch {epoch}).")
				break
				
			if self.scheduler is not None:
				self.scheduler.step()
	
		if self.best_state is not None:
			self.model.load_state_dict(self.best_state['model'])
			print(f"Loaded best model from epoch {self.best_state['epoch']} with loss {self.best_state['loss']:.6f}")
		
		self.visualise(lr, hr)
	
	def visualise(self, lr, hr, out=None):
		if (out is None):
			self.model.eval()
			with torch.no_grad():
				out = self.model(self.model.z)
				out = out.cpu().clamp(0, 1)

		fig, axes = plt.subplots(1, 3, figsize=(60,20))

		axes[0].imshow(lr.squeeze(0).permute(1,2,0).float())
		axes[0].set_title("Low Resolution")

		axes[1].imshow(out.squeeze(0).permute(1,2,0).float())
		axes[1].set_title("DIP Output")
		
		axes[2].imshow(hr.squeeze(0).permute(1,2,0).float())
		axes[2].set_title("High Resolution")

		for a in axes:
			a.axis("off")
		
		os.makedirs("results", exist_ok=True)
		plt.savefig("results/best.png")
		plt.close(fig)