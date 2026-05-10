import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F


#defining model
#今回は3層のMLP

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(30,40)
        self.fc2 = nn.Linear(40,35)
        self.fc3 = nn.Linear()

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)#使うハードウェアの確認

net = MLP().to(device)

criterion = nn.CrossEntropyLoss()#誤差関数をクロスエントロピーで
optimizer = optim.SGD(net.parameters(), lr = 0.01)#勾配降下で最適化
