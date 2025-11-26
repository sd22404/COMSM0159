import torch
import torch.nn as nn
import numpy as np
import math

def add_module(self, module):
	self.add_module(str(len(self) + 1), module)
	
torch.nn.Module.add = add_module

def bn(num_features):
	return nn.BatchNorm2d(num_features)

def act(act_fun = 'LeakyReLU'):
	'''
		Either string defining an activation function or module (e.g. nn.ReLU)
	'''
	if isinstance(act_fun, str):
		if act_fun == 'LeakyReLU':
			return nn.LeakyReLU(0.2, inplace=True)
		elif act_fun == 'Swish':
			assert False
		elif act_fun == 'ELU':
			return nn.ELU()
		elif act_fun == 'none':
			return nn.Sequential()
		else:
			assert False
	else:
		return act_fun()

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
	downsampler = None
	if stride != 1 and downsample_mode != 'stride':

		if downsample_mode == 'avg':
			downsampler = nn.AvgPool2d(stride, stride)
		elif downsample_mode == 'max':
			downsampler = nn.MaxPool2d(stride, stride)
		elif downsample_mode  in ['lanczos2', 'lanczos3']:
			assert False
		else:
			assert False

		stride = 1

	padder = None
	to_pad = int((kernel_size - 1) / 2)
	if pad == 'reflection':
		padder = nn.ReflectionPad2d(to_pad)
		to_pad = 0
  
	convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


	layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
	return nn.Sequential(*layers)

class Concat(nn.Module):
	def __init__(self, dim, *args):
		super(Concat, self).__init__()
		self.dim = dim

		for idx, module in enumerate(args):
			self.add_module(str(idx), module)

	def forward(self, input):
		inputs = []
		for module in self._modules.values():
			inputs.append(module(input))

		inputs_shapes2 = [x.shape[2] for x in inputs]
		inputs_shapes3 = [x.shape[3] for x in inputs]        

		if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
			inputs_ = inputs
		else:
			target_shape2 = min(inputs_shapes2)
			target_shape3 = min(inputs_shapes3)

			inputs_ = []
			for inp in inputs: 
				diff2 = (inp.size(2) - target_shape2) // 2 
				diff3 = (inp.size(3) - target_shape3) // 2 
				inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

		return torch.cat(inputs_, dim=self.dim)

	def __len__(self):
		return len(self._modules)


def skip(num_input_channels=3, num_output_channels=3,
		num_channels_down=[64, 64, 128, 128, 128], num_channels_up=[64, 64, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
		filter_size_down=3, filter_size_up=3, filter_skip_size=1,
		need_sigmoid=True, need_bias=True, 
		pad='zero', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', 
		need1x1_up=True):
	"""Assembles encoder-decoder with skip connections.

	Arguments:
		act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
		pad (string): zero|reflection (default: 'zero')
		upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
		downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

	"""
	
	assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

	n_scales = len(num_channels_down) 

	if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
		upsample_mode   = [upsample_mode]*n_scales

	if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
		downsample_mode   = [downsample_mode]*n_scales
	
	if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
		filter_size_down   = [filter_size_down]*n_scales

	if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
		filter_size_up   = [filter_size_up]*n_scales

	last_scale = n_scales - 1 

	cur_depth = None

	model = nn.Sequential()
	model_tmp = model

	input_depth = num_input_channels
	for i in range(len(num_channels_down)):

		deeper = nn.Sequential()
		skip = nn.Sequential()

		if num_channels_skip[i] != 0:
			model_tmp.add(Concat(1, skip, deeper))
		else:
			model_tmp.add(deeper)
		
		model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

		if num_channels_skip[i] != 0:
			skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
			skip.add(bn(num_channels_skip[i]))
			skip.add(act(act_fun))
			
		# skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

		deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
		deeper.add(bn(num_channels_down[i]))
		deeper.add(act(act_fun))

		deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
		deeper.add(bn(num_channels_down[i]))
		deeper.add(act(act_fun))

		deeper_main = nn.Sequential()

		if i == len(num_channels_down) - 1:
			# The deepest
			k = num_channels_down[i]
		else:
			deeper.add(deeper_main)
			k = num_channels_up[i + 1]

		deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

		model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
		model_tmp.add(bn(num_channels_up[i]))
		model_tmp.add(act(act_fun))


		if need1x1_up:
			model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
			model_tmp.add(bn(num_channels_up[i]))
			model_tmp.add(act(act_fun))

		input_depth = num_channels_down[i]
		model_tmp = deeper_main

	model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
	if need_sigmoid:
		model.add(nn.Sigmoid())

	return model


class UNet(nn.Module):
	def __init__(self, height, width, out_channels=3, base_channels=32, z_channels=32, z_sigma=0.05):
		super(UNet, self).__init__()
		self.out_h = height
		self.out_w = width
		self.out_c = out_channels
		self.bc = base_channels
		self.zc = z_channels
		self.std = z_sigma

		# ensure divisible by 16 to fit architecture
		pad_h = math.ceil(height / 16) * 16
		pad_w = math.ceil(width / 16) * 16
		
		self.register_buffer('z', torch.randn((1, self.zc, pad_h, pad_w)) * 0.1)

		print(f"UNet initialized with input shape: ({self.zc}, {self.out_h}, {self.out_w}) -> Padded: ({pad_h}, {pad_w})")
		
		self.input = unet_conv(self.zc, self.bc)

		self.down1 = down_conv(self.bc, self.bc * 2)
		self.down2 = down_conv(self.bc * 2, self.bc * 4)
		self.down3 = down_conv(self.bc * 4, self.bc * 8)
		self.down4 = down_conv(self.bc * 8, self.bc * 16)

		self.up4 = up_conv(self.bc * 16, self.bc * 8)
		self.up3 = up_conv(self.bc * 8, self.bc * 4)
		self.up2 = up_conv(self.bc * 4, self.bc * 2)
		self.up1 = up_conv(self.bc * 2, self.bc)

		self.output = out_conv(self.bc, self.out_c)

	def get_z(self, sigma=0.0):
		if sigma <= 0:
			return self.z
		return self.z + sigma * torch.randn_like(self.z)

	def forward(self, z=None, t=None, c=None):
		z = self.get_z(self.std) if z is None else z
		z = torch.cat((c, z), dim=1) if c is not None else z
		
		x = self.input(z)

		s1 = x
		x = self.down1(x)
		s2 = x
		x = self.down2(x)
		s3 = x
		x = self.down3(x)
		s4 = x
		x = self.down4(x)

		x = self.up4(x, s4)
		x = self.up3(x, s3)
		x = self.up2(x, s2)
		x = self.up1(x, s1)

		x = self.output(x)

		# crop back to original HR size
		x = x[:, :, :self.out_h, :self.out_w]
		return x


class unet_conv(nn.Module):
	def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
		super(unet_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride, padding, padding_mode='reflect'),
		)

	def forward(self, x):
		return self.conv(x)

class double_conv(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			conv(c_in, c_out, kernel_size, stride, padding),
			nn.LeakyReLU(0.2, inplace=True),
			conv(c_out, c_out, kernel_size, stride, padding),
			nn.LeakyReLU(0.2, inplace=True)
		)
	
	def forward(self, x):
		return self.conv(x)

class down_conv(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
		super(down_conv, self).__init__()
		self.down = nn.Sequential(
			# nn.MaxPool2d(2, 2),
			conv(c_in, c_out, kernel_size=2, stride=2, padding=0),
			double_conv(c_out, c_out, kernel_size, stride, padding)
		)

	def forward(self, x):
		return self.down(x)
	
class up_conv(nn.Module):
	def __init__(self, c_in, c_out):
		super(up_conv, self).__init__()
		# self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
		)
		self.conv = double_conv(c_in, c_out)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat((x, skip), dim=1)
		return self.conv(x)

class out_conv(nn.Module):
	def __init__(self, in_c, out_c):
		super(out_conv, self).__init__()
		self.conv = nn.Sequential(
			conv(in_c, out_c, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.conv(x)