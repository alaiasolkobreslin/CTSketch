import torch
import torch.nn.functional as F
from torch.nn.functional import conv1d

# Grid size and moves
grid_size = 12
max_steps = 24
num_moves = 8

# Define move offsets
directions = {
    "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
    "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)
}

# Define perception network outputs (logits) for a 12x12 grid
logits = torch.randn(12, 12, 5)  # Example logits
tile_distributions = F.softmax(logits, dim=-1)  # Probabilities over 5 costs

# Initialize edge distributions for 8 directions
# Add small padding to simplify neighbor indexing
tile_distributions_padded = F.pad(tile_distributions, (0, 0, 1, 1, 1, 1), value=0)
# Extract edge distributions for each direction
edge_distributions = {}
for direction, (di, dj) in directions.items():
    shifted = tile_distributions_padded[1 + di : 13 + di, 1 + dj : 13 + dj, :]
    edge_distributions[direction] = shifted

# Move probability tensor (24 steps, 8 moves)
move_probabilities = torch.zeros(max_steps, num_moves)

# Initial path cost distribution (12x12 grid, 5 cost classes)
path_cost_distribution = torch.zeros(grid_size, grid_size, 5)

def convolve_costs(cost_dist1, cost_dist2):
    """
    Convolve two cost distributions element-wise for all nodes.
    Args:
        cost_dist1: Tensor of shape (12, 12, len(cost_bins))
        cost_dist2: Tensor of shape (12, 12, len(cost_bins))
    Returns:
        Tensor of shape (12, 12, len(cost_bins))
    """
    _, grid_size, _ = cost_dist1.shape
    result = torch.zeros_like(cost_dist1)

    # Iterate over all grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            dist1 = cost_dist1[i, j].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, len(cost_bins))
            dist2 = cost_dist2[i, j].unsqueeze(0).unsqueeze(0)  # Same shape
            conv_result = F.conv1d(dist1, dist2, padding="same")
            result[i, j] = conv_result.squeeze()
    
    return result

# Initialize step-by-step tracking of moves
for step in range(max_steps):
    new_path_cost_distribution = torch.zeros_like(path_cost_distribution)
    move_contributions = torch.zeros(grid_size, grid_size, num_moves)

    for move_idx, (direction, (di, dj)) in enumerate(directions.items()):
        # Get the edge cost distribution for the current move
        edge_distribution = edge_distributions[direction]

        # Shift the current path cost distribution according to the move direction
        shifted_path_cost = F.pad(path_cost_distribution, (0, 0, 1, 1, 1, 1), value=0)
        shifted_path_cost = shifted_path_cost[1 + di : 1 + grid_size + di, 1 + dj : 1 + grid_size + dj, :]

        # Compute the contribution of this move
        contribution = convolve_costs(shifted_path_cost, edge_distribution)
        new_path_cost_distribution += contribution
        move_contributions[..., move_idx] = torch.sum(contribution, dim=-1)

    # Normalize move contributions to compute probabilities
    move_probabilities[step] = torch.sum(move_contributions, dim=(0, 1))
    move_probabilities[step] /= move_probabilities[step].sum()  # Normalize

    # Update path cost distribution for the next step
    path_cost_distribution = new_path_cost_distribution

print(path_cost_distribution)