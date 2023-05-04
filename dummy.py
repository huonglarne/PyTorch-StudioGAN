import torch

x = torch.load('x.pt').cuda()
y = torch.load('y.pt').cuda()

torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))