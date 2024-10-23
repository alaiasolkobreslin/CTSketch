import torch

from tensorsketch.tensor_approx import TensorApprox
from tt_sketch.sketch import hmt_sketch
import tt_sketch

class TensorSketch():
    def __init__(self, method):
        super().__init__()
        self.method = method

    def approx_theta(self, configs):
        gt = configs['gt'] if 'gt' in configs else None
        digit = configs['digit'] if 'gt' in configs else None
        
        if self.method == 'tt':
            return self.tt_approx(gt, digit)
        elif self.method == 'hooi':
            return self.hooi_approx(gt, digit)
        else: raise Exception(f"{self.method} not implemented.")

    def tt_approx(self, gt, digits):
        tensor_sum = tt_sketch.tensor.DenseTensor(gt.numpy())
        tt_sketched = hmt_sketch(tensor_sum, 2)
        rerr = tt_sketched.error(tensor_sum, relative=True)
        X_hat = torch.from_numpy(tt_sketched.to_numpy()).round()
        X_hat = torch.minimum(torch.maximum(X_hat, torch.zeros_like(X_hat)), torch.ones_like(X_hat)*(digits*9))
        return rerr, 

    def hooi_approx(self, gt, digit):
        X = gt.cpu().numpy()
        tapprox1 = TensorApprox(X, [2]*digit, [5]*digit, [11]*digit)
        X_hat, core_sketch, arm_sketches, rerr, (sketch_time, recover_time) = tapprox1.tensor_approx("twopass") 
        X_hat = torch.from_numpy(X_hat).round()
        X_hat = torch.minimum(torch.maximum(X_hat, torch.zeros_like(X_hat)), torch.ones_like(X_hat)*(digit*9))
        return rerr, X_hat.long()