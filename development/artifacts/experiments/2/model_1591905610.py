#from,to=2,10
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

class NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(784, 128)
		self.output = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(x.size()[0], -1)
		x = self.hidden(x)
		x = F.sigmoid(x)
		x = self.output(x)
		return x

criterion = nn.CrossEntropyLoss()

model = NN()

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)

