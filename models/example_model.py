import torch
import torch.nn as nn
import torch.nn.functional as F

class ExampleModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.some_parameter = nn.Parameter(torch.randn(1))

	def forward(self, x):
		return x