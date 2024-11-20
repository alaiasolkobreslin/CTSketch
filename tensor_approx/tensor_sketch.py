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
        digit = configs['digit'] if 'digit' in configs else None
        rank = configs['rank'] if 'rank' in configs else 2
        
        if self.method == 'tt':
            return self.tt_approx(gt, rank)
        elif self.method == 'hooi' or self.method == 'onepass' or self.method == 'twopass' :
            return self.hooi_approx(gt, digit, rank)
        else: raise Exception(f"{self.method} not implemented.")

    def tt_approx(self, gt, rank):
        tensor_sum = tt_sketch.tensor.DenseTensor(gt)
        tt_sketched = hmt_sketch(tensor_sum, rank)
        rerr = tt_sketched.error(tensor_sum, relative=True)
        X_hat = tt_sketched.to_numpy().round()
        return rerr, tt_sketched.cores, X_hat

    def hooi_approx(self, gt, digits, rank):
        X = gt.float()
        tapprox1 = TensorApprox(X, [rank]*digits, [rank]*digits, [rank*2 + 1]*digits)
        X_hat, core_sketch, arm_sketches, rerr, (sketch_time, recover_time) = tapprox1.tensor_approx(self.method) 
        X_hat = torch.clamp(X_hat, 0, digits*9).long()
        return rerr, [core_sketch] + arm_sketches, X_hat