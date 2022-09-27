import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ConditionZeroDCE(nn.Module):

	def __init__(self):
		super(ConditionZeroDCE, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		ch_n = 256
		self.e_conv1 = nn.Conv2d(1, ch_n, 3, 1, 1, bias=True) 
		self.e_conv2 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True) 
		self.e_conv3 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True) 
		self.e_conv4 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True) 
		self.e_conv5 = nn.Conv2d(ch_n*3, ch_n, 3, 1, 1, bias=True) 
		self.e_conv6 = nn.Conv2d(ch_n*3, ch_n, 3, 1, 1, bias=True) 
		self.e_conv7 = nn.Conv2d(ch_n*3, 10, 3, 1, 1, bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

		self.e_conv_e1 = nn.Conv2d(1, ch_n, 3, 1, 1, bias=True) 
		self.e_conv_e2 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True) 
		self.e_conv_e3 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True) 
		self.e_conv_e4 = nn.Conv2d(ch_n, ch_n, 3, 1, 1, bias=True)  
   
	def forward(self, x, e):
		E_x = e.repeat(x.shape[0],1,1,1) 
		E_x1 = self.relu(self.e_conv_e1(E_x))
		E_x2 = self.relu(self.e_conv_e2(E_x1))
		E_x3 = self.relu(self.e_conv_e3(E_x2))
		E_x4 = self.relu(self.e_conv_e4(E_x3))
		

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([E_x4, x3, x4], 1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([E_x3, x2, x5], 1)))

		x_r = torch.tanh(self.e_conv7(torch.cat([E_x2, x1, x6], 1)))
		r1,r2,r3,r4,r5,r6,r7,r8,r9,r10 = torch.split(x_r, 1, dim=1)

		x = x + r1*(torch.pow(x, 2) - x)
		x = x + r2*(torch.pow(x, 2) - x)
		x = x + r3*(torch.pow(x, 2) - x)
		enhance_image_1 = x + r4*(torch.pow(x, 2) - x)     
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1, 2) - enhance_image_1)     
		x = x + r6*(torch.pow(x, 2) - x)   
		x = x + r7*(torch.pow(x, 2)-x)
		enhance_image_2 = x + r8*(torch.pow(x, 2) - x)
		x = enhance_image_2 + r9*(torch.pow(enhance_image_2, 2) - enhance_image_2) 
		enhance_image= x + r10*(torch.pow(x, 2) - x)

		return enhance_image
    