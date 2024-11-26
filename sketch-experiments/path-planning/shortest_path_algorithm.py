import torch

def shortest_path(weights):
    n = weights.size(0)
    # Initialize distance matrix
    dist = weights.clone()
    
    # iterate for paths up to length n-1
    for _ in range(n-1):
        dist = torch.min(dist.unsqueeze(2) + weights.unsqueeze(0), dim=1)[0]
    return dist
    
W = torch.tensor([[0, 3, float('inf')],
                  [float('inf'), 0, 1],
                  [1, float('inf'), 0]])

D = shortest_path(W)
print(D)