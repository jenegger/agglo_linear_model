import torch
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from my_first_model import TinyModel
from matplotlib import pyplot as plt

class CustomDataset(Dataset):
	def __init__(self):
		xy = np.loadtxt("combined_stuff_test.txt",delimiter=",",dtype=np.float32)
		self.x = torch.from_numpy(xy[:,:-1])
		self.y = torch.from_numpy(xy[:,[-1]])
		self.n_samples = xy.shape[0]
	def __len__(self):
		return self.n_samples
	def __getitem__(self, idx):
		return self.x[idx],self.y[idx]
#define model
model = TinyModel()
#define loss,lr, etc...
learning_rate = 0.01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate)

dataset = CustomDataset()
batches = 256
dloader = DataLoader(dataset,batch_size=batches,shuffle=True)
dataiter= iter(dloader)
data = dataiter.next()
features,labels= data
print(features,labels)
#training loop
num_epochs = 50
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batches)
print(total_samples)
print(n_iterations)
l_loss = []
for  epoch in range(num_epochs):
	for i, (inputs,labels) in enumerate(dloader):
		#forward and backward, update
		output = model(inputs)
		#print("this is the output")
		#print(output)
		#print("this is the label")
		#print(labels)
		l = loss(output,labels)
		l.backward()
		optimizer.step()
		optimizer.zero_grad()
		#print(l.item())
		l_loss.append(l.item())
		#if epoch  % 10 == 0:
		#	print("loss = ",l.item())
		#	l_loss.append(l.item())
	
plt.plot(l_loss)
plt.show()
