from time import time
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class DIPTrainer:
	def __init__(self, model, criterion, optimizer, train_loader, device):
		self.train_loader = train_loader
		self.device = device
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.noise = torch.randn(1, 3, 64, 64).to(device)

	def train(self, epochs):
		self.model.train()
		self.low_res, self.high_res = next(iter(self.train_loader))
		self.low_res = self.low_res.to(self.device)
		for epoch in range(epochs):
			epoch_start_time = time()
			self.optimizer.zero_grad()

			output = self.model(self.noise)
			lr_output = F.resize(output, size=self.low_res.shape[2:])
			loss = self.criterion(lr_output, self.low_res)
			loss.backward()
			self.optimizer.step()

			epoch_train_time = time() - epoch_start_time
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.6f}, Time: {epoch_train_time:.2f} seconds")
	
	def visualise(self):
		self.model.eval()
		with torch.no_grad():
			output = self.model(self.noise)
		output = output.cpu()
		fig = plt.figure()
		ax1 = fig.add_subplot(2,2,0)
		ax2 = fig.add_subplot(2,2,1)
		ax3 = fig.add_subplot(2,2,2)
		ax1.imshow(output.squeeze().permute(1,2,0))
		ax1.set_title("DIP Output")
		ax2.imshow(self.low_res.cpu().squeeze().permute(1,2,0))
		ax2.set_title("Low Resolution")
		ax3.imshow(self.high_res.squeeze().permute(1,2,0))
		ax3.set_title("High Resolution")
		plt.savefig("dip_out.png")



class DiffusionTrainer:
	def __init__(self, model, criterion, optimizer, train_loader, val_loader, device):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer

	def train(self, epochs):
		# TODO: Set model to train mode
		self.model.train()
		# Initialise gradient scaler for mixed precision training
		scaler = torch.cuda.amp.GradScaler()
		# TODO: Define loss function (Mean Squared Error Loss)
		mse_loss = torch.nn.MSELoss()

		total_train_time = 0
		for epoch in range(epochs):
			epoch_start_time = time()
			total_loss = 0.0
			for batch_idx, (x, _) in enumerate(self.train_loader):
				# Clear previous gradients
				self.optimizer.zero_grad()
				x = x.to(self.device)

				# Forward pass with mixed precision
				# - **Mixed Precision**: `torch.cuda.amp.autocast()` enables faster training with less memory usage.
				with torch.cuda.amp.autocast():
					# TODO: Call the diffusion model to get noisy input
					noisy, e, pred_e = self.model.forward(x)
					# TODO: Calculate the denoising loss
					denoising_loss = mse_loss(pred_e, e)
				# Use the scaler to scale the loss and backpropagate
				scaler.scale(denoising_loss).backward()

				# TODO: Update the model parameters using the optimizer
				scaler.step(self.optimizer)
				# TODO: Update the scaler for the next iteration
				scaler.update()
				# TODO: Accumulate the loss
				total_loss += denoising_loss.item()

			# TODO: Print the average loss for this epoch
			avg_loss = total_loss / len(self.train_loader)
			print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

			# TODO: Print the training time for this epoch
			epoch_train_time = time() - epoch_start_time
			total_train_time += epoch_train_time
			print(f"Epoch training time: {epoch_train_time:.2f} seconds")

		print("Training finished!")
		print(f"Total training time: {total_train_time:.2f} seconds")