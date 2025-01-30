from tt_sketch.sketch import hmt_sketch
import tt_sketch

class TensorSketch():
    def __init__(self, method):
        super().__init__()
        self.method = method

    def approx_theta(self, configs):
        gt = configs['gt'] if 'gt' in configs else None
        rank = configs['rank'] if 'rank' in configs else 2
        
        if self.method == 'tt':
            return self.tt_approx(gt, rank)
        else: raise Exception(f"{self.method} not implemented.")

    def tt_approx(self, gt, rank):
        tensor_sum = tt_sketch.tensor.DenseTensor(gt)
        tt_sketched = hmt_sketch(tensor_sum, rank)
        rerr = tt_sketched.error(tensor_sum, relative=True)
        return rerr, tt_sketched.cores, tt_sketched