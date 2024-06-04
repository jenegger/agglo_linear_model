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
		#xy = np.loadtxt("combined_stuff_test.txt",delimiter=",",dtype=np.float32)
		xy = np.loadtxt("comb_events.txt",delimiter=",",dtype=np.float32)
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
#learning_rate = 0.005 #old lr
learning_rate = 8e-5

loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate) #old optimizer
#optimizer = torch.optim.Adam(model.parameters(),lr= learning_rate)

dataset = CustomDataset()
batches = 64
dloader = DataLoader(dataset,batch_size=batches,shuffle=True)
dataiter= iter(dloader)
data = next(dataiter)
features,labels= data
print(features,labels)
#training loop
num_epochs = 100
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batches)
print(total_samples)
print(n_iterations)
l_loss = []
for  epoch in range(num_epochs):
	for i, (inputs,labels) in enumerate(dloader):
		#forward and backward, update
		output = model(inputs)
		l = loss(output,labels)
		l.backward()
		optimizer.step()
		#optimizer.zero_grad()
		l_loss.append(l.item())
plt.title("Loss functions")	
plt.xlabel("Iterations")
plt.ylabel("Loss")
#plt.yscale('log')
plt.plot(l_loss)
plt.show()
