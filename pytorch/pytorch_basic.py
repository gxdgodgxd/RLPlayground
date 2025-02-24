import torch

x = torch.empty(5, 3)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)

print(x.size())
print(x[:, 1])
print(x[1, :])
print(x[1, 1])


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1的意思是让PyTorch自己推断出第一维的大小。
a = x.view(2,2,4)
print(x.size(), y.size(), z.size(), a.size())

x = torch.randn(1)
print(x)
print(x.item())

