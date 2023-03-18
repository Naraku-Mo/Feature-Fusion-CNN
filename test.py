import torch

n=torch.FloatTensor(2,3).fill_(1)
print("before:",n)
n[0:1,1:2]=0
n[1:,0:1]=0
print("after:",n)