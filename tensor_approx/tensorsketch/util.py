import numpy as np
import tensorly as tl
import torch

tl.set_backend('pytorch')

class TensorInfoBucket(object):
    def __init__(self, tensor_shape, ks, ranks, ss = []):
        '''
        Information of the original tensor X
        :k,s: integer
        :ranks: n-darray for the ranks of X
        '''
        self.tensor_shape = tensor_shape
        self.ks = ks
        self.ranks = ranks
        self.ss = ss

    def get_info(self):
        return self.tensor_shape, self.ks, self.ranks, self.ss

class RandomInfoBucket(object):
    ''' 
    Information for generating randomized linear maps
    ''' 
    def __init__(self, std=1, typ='g', random_seed = 0, sparse_factor = 0.1):
        self.std = std
        self.typ = typ
        self.random_seed = random_seed
        self.sparse_factor = sparse_factor

    def get_info(self):
        return self.std, self.typ, self.random_seed, self.sparse_factor

def random_matrix_generator(m, n, Rinfo_bucket):

    std, typ, random_seed, sparse_factor = Rinfo_bucket.get_info()
    torch.manual_seed(random_seed)
    types = set(['g', 'u', 'sp'])
    assert typ in types, "please aset your type of random variable correctly"

    if typ == 'g':
        return torch.randn((m, n))*std
    elif typ == 'u':
        return (torch.rand((m, n))*2 - 1)*torch.sqrt(torch.tensor(3))*std
    elif typ == 'sp':
        return torch.distributions.Binomial(1, sparse_factor).sample((m, n))*\
        torch.where(torch.randint(-1, 1, (2,3)) == 0, 1, -1)*torch.sqrt(3)*std
    elif typ == 'ssrft': 
        return 0

def tensor_gen_help(core,arms):
    '''
    :param core: the core tensor in higher order svd s*s*...*s
    :param arms: those arms n*s
    :return:
    '''
    for i in torch.arange(len(arms)):
        prod = tl.tenalg.mode_dot(core,arms[i],mode =i)
    return prod 


def generate_super_diagonal_tensor(diagonal_elems, dim):
    n = len(diagonal_elems)
    tensor = torch.zeros(torch.Tensor(n).repeat(dim))
    for i in range(n):
        index = tuple([i for _ in range(dim)])
        tensor[index] = diagonal_elems[i]
    return tl.tensor(tensor)



def square_tensor_gen(n, r, dim = 3,  typ = 'id', noise_level = 0, seed = None):
    '''
    :param n: size of the tensor generated n*n*...*n
    :param r: rank of the tensor or equivalently, the size of core tensor
    :param dim: # of dimensions of the tensor, default set as 3
    :param typ: identity as core tensor or low rank as core tensor
    :param noise_level: sqrt(E||X||^2_F/E||error||^_F)
    :return: The tensor with noise, and The tensor without noise
    '''
    if seed: 
        torch.manual_seed(seed)

    types = set(['id', 'lk', 'fpd', 'spd', 'sed', 'fed'])
    assert typ in types, "please set your type of tensor correctly"
    total_num = torch.pow(n, dim)

    if typ == 'id':
        elems = [1 for _ in range(r)]
        elems.extend([0 for _ in range(n-r)])
        noise = torch.randn([n for _ in range(dim)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0 +noise*torch.sqrt((noise_level**2)*r/total_num), X0
        
    if typ == 'spd':
        elems = [1 for _ in range(r)]
        elems.extend([1.0/i for i in range(2, n-r+2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == 'fpd':
        elems = [1 for _ in range(r)]
        elems.extend([1.0/(i*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'sed':
        elems = [1 for _ in range(r)]
        elems.extend([torch.pow(10, -0.25*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'fed':
        elems = [1 for _ in range(r)]
        elems.extend([torch.pow(10, (-1.0)*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == "lk":
        core_tensor = torch.rand([r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in torch.arange(dim):
            arm = torch.randn((n,r))
            arm, _ = torch.linalg.qr(arm)
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
        true_signal_mag = torch.linalg.norm(core_tensor)**2
        noise = torch.randn(torch.Tensor(n).repeat(dim))
        X = tensor + noise*torch.sqrt((noise_level**2)*true_signal_mag/torch.prod\
            (total_num))
        return X, tensor

def eval_rerr(X,X_hat,X0):
    error = X-X_hat
    return torch.linalg.norm(error.reshape(-1 ,1),'fro')/ \
    torch.linalg.norm(X0.reshape(-1,1),'fro')

if __name__ == "__main__":
    tl.set_backend('pytorch')
    X = square_tensor_gen(5, 3, dim=3, typ='id', noise_level=0.1)
