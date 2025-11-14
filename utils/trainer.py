class Trainer:
	def __init__(self, model, criterion, optimizer, train_loader, val_loader, device):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device

	def train(self, num_epochs):
		return NotImplementedError("Training loop not implemented yet.")