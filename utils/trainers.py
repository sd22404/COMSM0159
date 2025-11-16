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


class GANTrainer(Trainer):
	def __init__(self, generator, discriminator, g_criterion, d_criterion, g_optimizer, d_optimizer, train_loader, val_loader, device):
		super(GANTrainer, self).__init__(generator, g_criterion, g_optimizer, train_loader, val_loader, device)
		self.discriminator = discriminator
		self.d_criterion = d_criterion
		self.d_optimizer = d_optimizer

	def train(self, num_epochs):
		return NotImplementedError("GAN training loop not implemented yet.")