#!/usr/bin/env python3
import torch

class TinyModel(torch.nn.Module):
	def __init__(self):
		super(TinyModel, self).__init__()
		self.linear1 = torch.nn.Linear(11,64)
		self.activation = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(64,1)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.sigmoid(x)
		return x

#tinymodel = TinyModel()
