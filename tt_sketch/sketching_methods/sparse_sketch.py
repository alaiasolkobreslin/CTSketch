from typing import Optional, Tuple, Union

import torch
from tt_sketch.tensor import SparseTensor

def _Psi_core_slice(
    inds,
    entries,
    left_sketch_vec,
    right_sketch_vec,
    mu: int,
    j: int,
):
    """Compute slice of tensor Y[:,j,:] from left+right sketching matrix
    and entries+index of sparse tensor"""
    mask = inds[mu] == j

    if mu == 0:  # only sketch on right
        if right_sketch_vec is None:
            raise ValueError
        Psi_slice = entries[mask] @ right_sketch_vec[:, mask].T.float()
        Psi_slice = Psi_slice.reshape(1, -1)
    elif mu == inds.shape[0] - 1:  # only sketch on left
        if left_sketch_vec is None:
            raise ValueError
        Psi_slice = left_sketch_vec[:, mask].float() @ entries[mask]
        Psi_slice = Psi_slice.reshape(-1, 1)
    else:  # sketch on both sides
        if left_sketch_vec is None or right_sketch_vec is None:
            raise ValueError
        Psi_slice = (
            left_sketch_vec[:, mask] * entries[mask]
        ) @ right_sketch_vec[:, mask].T
    return Psi_slice


def sketch_omega_sparse(
    left_sketch,
    right_sketch,
    *,
    tensor: SparseTensor,
    **kwargs,
):
    return (left_sketch * tensor.entries) @ right_sketch.T


def sketch_psi_sparse(
    left_sketch,
    right_sketch,
    *,
    tensor: SparseTensor,
    mu: int,
    psi_shape: Tuple[int, int, int],
    **kwargs,
):
    Psi = torch.zeros(psi_shape)
    n_mu = psi_shape[1]
    for j in range(n_mu):
        Psi[:, j, :] = _Psi_core_slice(
            tensor.indices,
            tensor.entries,
            left_sketch,
            right_sketch,
            mu,
            j,
        )
    return Psi
