import torch

from tensorsketch.tensor_approx import TensorApprox
from tt_sketch.sketch import hmt_sketch
import tt_sketch

class TensorSketch():
    def __init__(self, method):
        super().__init__()
        self.method = method

    def approx_theta(self, configs):
        gt1 = configs['gt1'] if 'gt1' in configs else None
        gt2 = configs['gt2'] if 'gt2' in configs else None
        digit = configs['digit'] if 'gt1' in configs else None
        components = configs['components'] if 'gt1' in configs else None
        
        if self.method == 'tt':
            return self.tt_approx(gt1, gt2, digit, components)
        elif self.method == 'hooi':
            return self.hooi_approx(gt1, gt2, digit, components)
        else: raise Exception(f"{self.method} not implemented.")

    def tt_approx(self, gt, digits, components):
        raise("Not implemented with components")
        tensor_sum = tt_sketch.tensor.DenseTensor(gt.numpy())
        tt_sketched = hmt_sketch(tensor_sum, 2)
        rerr = tt_sketched.error(tensor_sum, relative=True)
        X_hat = torch.from_numpy(tt_sketched.to_numpy()).round()
        X_hat = torch.minimum(torch.maximum(X_hat, torch.zeros_like(X_hat)), torch.ones_like(X_hat)*(digits*9))
        return rerr, 

    def hooi_approx(self, gt1, gt2, digit, components):
        X1 = gt1.cpu().numpy()
        X2 = gt2.cpu().numpy()
        # X, ranks, ks, ss, where ks is the reduced dimension of the arm tensors
        tapprox1 = TensorApprox(X1, [2]*digit, [5]*digit, [11]*digit)
        X_hat1, _, _, rerr1, (_, _) = tapprox1.tensor_approx("twopass") 
        X_hat1 = torch.from_numpy(X_hat1).round()
        X_hat1 = torch.minimum(torch.maximum(X_hat1, torch.zeros_like(X_hat1)), torch.ones_like(X_hat1)*(digit*9))
        tapprox2 = TensorApprox(X2, [2]*components, [10]*components, [22]*components)
        X_hat2, _, _, rerr2, (_, _) = tapprox2.tensor_approx("twopass") 
        X_hat2 = torch.from_numpy(X_hat2).round()
        X_hat2 = torch.minimum(torch.maximum(X_hat2, torch.zeros_like(X_hat2)), torch.ones_like(X_hat2)*(digit*9))
        return rerr1, rerr2, X_hat1.long(), X_hat2.long()