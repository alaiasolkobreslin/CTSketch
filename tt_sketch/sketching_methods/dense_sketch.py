import torch
from tt_sketch.tensor import DenseTensor
from tt_sketch.utils import matricize


def sketch_omega_dense(
    left_sketch,
    rigth_sketch,
    *,
    tensor: DenseTensor,
    mu: int,
    **kwargs
):
    X_mat = matricize(tensor.data, range(mu + 1), mat_shape=True)
    return left_sketch @ X_mat @ rigth_sketch.T


def sketch_psi_dense(
    left_sketch,
    right_sketch,
    *,
    tensor: DenseTensor,
    mu: int,
    **kwargs
):
    ndim = tensor.ndim
    tensor_data = tensor.data
    if left_sketch is None:
        tensor_mat = matricize(tensor_data, 0, mat_shape=True)
        Psi = tensor_mat @ right_sketch.T.float()
        Psi = Psi.reshape((1,) + Psi.shape)
    elif right_sketch is None:
        tensor_mat = matricize(tensor_data, ndim - 1, mat_shape=True).T
        Psi = left_sketch.float() @ tensor_mat
        Psi = Psi.reshape(Psi.shape + (1,))
    else:
        # Compute order 3 tensorisation of X
        tensor_ord3 = matricize(tensor_data, range(mu + 1), mat_shape=False)
        tensor_ord3 = tensor_ord3.reshape(
            torch.prod(torch.tensor(tensor_ord3.shape[:mu]), dtype=int),
            tensor_ord3.shape[mu],
            tensor_ord3.shape[mu + 1],
        )
        Psi = torch.einsum(
            "ij,jkl,ml->ikm",
            left_sketch,
            tensor_ord3,
            right_sketch
        )
    return Psi
