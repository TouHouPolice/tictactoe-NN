
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,36)
        self.fc4 = nn.Linear(36,2)
        self.device=None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")
        self.to(self.device)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)

        return F.log_softmax(x,dim=1).to(self.device)