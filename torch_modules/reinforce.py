import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class REINFORCE(nn.Module):
    def __init__(self, bbox, n_samples, output_mapping):
        super().__init__()
        self.f = bbox
        self.k = n_samples
        self.output_mapping = output_mapping
        
    # REINFORCE forward function takes the ground truth as input as it will be
    # used in the reward function.
    def forward(self, gt, *x):
        batch_size = x[0].size(0)
        categorical_probs = [dist.Categorical(x_i) for x_i in x]
        
        # Sample inputs and collect results
        sampled_indices = [cp.sample_n(self.k) for cp in categorical_probs]
        results = [0] * self.k
        for k in range(self.k):
            sampled_inputs = [si[k] for si in sampled_indices]
            results[k] = self.f(*sampled_inputs)
            
        log_p_samples = [categorical_probs[i].log_prob(sampled_indices[i]).sum(dim=-1) for i in range(categorical_probs)]
