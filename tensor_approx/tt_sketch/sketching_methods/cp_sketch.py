import torch
from tt_sketch.tensor import CPTensor


def sketch_omega_cp(
    left_sketch, right_sketch, **kwargs
):
    return left_sketch.T @ right_sketch


def sketch_psi_cp(
    left_sketch,
    right_sketch,
    *,
    tensor: CPTensor,
    mu: int,
    **kwargs,
):
    cp_core = tensor.cores[mu]

    if left_sketch is None:
        Psi = torch.einsum("ji,il->jl", cp_core, right_sketch)
        Psi = Psi.reshape((1,) + Psi.shape)
    elif right_sketch is None:
        Psi = torch.einsum("il,kl->ik", left_sketch.T, cp_core)
        Psi = Psi.reshape(Psi.shape + (1,))
    else:
        Psi = torch.einsum(
            "ij,kj,jm->ikm",
            left_sketch.T,
            cp_core,
            right_sketch
        )
    return Psi
