import torch

def make_grid(height, width):
	# centered pixel coordinates
	step_y = 2 / height
	y = torch.linspace(-1 + step_y / 2, 1 - step_y / 2, steps=height)
	step_x = 2 / width
	x = torch.linspace(-1 + step_x / 2, 1 - step_x / 2, steps=width)
	grid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
	return grid

def make_cells(coords, height, width):
	cells = torch.ones_like(coords)
	cells[:, 0] *= 2 / height
	cells[:, 1] *= 2 / width
	
	return cells