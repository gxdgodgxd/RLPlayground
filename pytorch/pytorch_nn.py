import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 32x32 -> 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 14x14 -> 10x10 -> 5x5
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了batch维度之外的其它维度。
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[2].size())


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
print(out.size())

target = torch.arange(1, 11, dtype=torch.float32)
target = target.view(1, -1)
print(target)
print(target[0][0].dtype)

criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

# net.zero_grad()     # 清掉tensor里缓存的梯度值。

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()

