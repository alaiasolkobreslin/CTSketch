import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ISED(nn.Module):
    def __init__(self, bbox, n_samples, semiring, output_mapping):
        super().__init__()
        self.f = bbox
        self.k = n_samples
        self.semiring = semiring
        self.output_mapping = output_mapping
        
    def forward(self, *x):
        batch_size = x[0].size(0)
        categorical_probs = [dist.Categorical(x_i) for x_i in x]
        
        # Sample inputs and collect results
        sampled_indices = [cp.sample_n(self.k) for cp in categorical_probs]
        results = [0] * self.k
        for k in range(self.k):
            sampled_inputs = [si[k] for si in sampled_indices]
            results[k] = self.f(*sampled_inputs)
        
        # Aggregate probabilities
        result_probs = torch.ones((batch_size, self.k))
        for (input, idx) in zip(x, sampled_indices):
            expanded_input = input.unsqueeze(0).expand(self.k, -1, -1)
            gathered_probs = torch.gather(expanded_input, 2, idx.unsqueeze(-1)).squeeze(-1).t()
            if self.semiring == "add-mult":
                result_probs *= gathered_probs
            elif self.semiring == "min-max":
                result_probs = torch.minimum(
                    result_probs.clone(), gathered_probs)
            else:
                raise Exception("Unknown semiring")
        
        # Vectorize result
        result_tensor = torch.zeros((batch_size, self.output_mapping))
        for i in range(batch_size):
            for k in range(self.k):
                if self.semiring == "add-mult":
                    result_tensor[i, results[k][i]] += result_probs[i, k]
                elif self.semiring == "min-max":
                    result_tensor[i, results[k][i]] = torch.maximum(
                        result_tensor[i, results[k][i]], result_probs[i, k])
                else:
                    raise Exception("Unknown semiring")
        y_pred = torch.nn.functional.normalize(result_tensor, dim=1)
        return y_pred