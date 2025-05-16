from tt_sketch.sketch import hmt_sketch
import tt_sketch

import tensorly as tl
from tensorly.decomposition import tucker, tensor_ring, parafac
tl.set_backend('pytorch')
import torch

class TensorSketch():
    def __init__(self, method):
        super().__init__()
        self.method = method

    def approx_theta(self, configs):
        gt = configs['gt'] if 'gt' in configs else None
        rank = configs['rank'] if 'rank' in configs else 2
        
        if self.method == 'tt':
            return self.tt_approx(gt, rank)
        elif self.method == 'tucker':
            return self.tucker_approx(gt, rank)
        elif self.method == 'cp':
            return self.cp_approx(gt, rank)
        elif self.method == 'tensor_ring':
            return self.tensor_ring_approx(gt, rank)
        else: raise Exception(f"{self.method} not implemented.")

    def tt_approx(self, gt, rank):
        tensor_sum = tt_sketch.tensor.DenseTensor(gt)
        tt_sketched = hmt_sketch(tensor_sum, rank)
        rerr = tt_sketched.error(tensor_sum, relative=True)
        return rerr, tt_sketched.cores, tt_sketched
    
    def cp_approx(self, gt, rank):
        X = gt.float()
        cp_decomp = parafac(X, rank=rank)
        X_hat =  tl.cp_to_tensor(cp_decomp)
        rerr = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
        return rerr, cp_decomp[1] + [cp_decomp[0]], X_hat


    def tucker_approx(self, gt, rank):
        X = gt.float()
        tucker_decomp = tucker(X, rank=rank)
        X_hat = tl.tucker_to_tensor(tucker_decomp) 
        rerr = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
        print(rerr)
        return rerr, [tucker_decomp.core] + tucker_decomp.factors, X_hat

    def tensor_ring_approx(self, gt, rank):
        X = gt.float()
        tr_decomp = tensor_ring(X, rank=rank)
        X_hat =  tl.tr_to_tensor(tr_decomp)
        rerr = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
        print(rerr)
        return rerr, tr_decomp.factors, X_hat