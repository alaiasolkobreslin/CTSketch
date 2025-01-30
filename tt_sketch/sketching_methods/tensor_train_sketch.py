from typing import Optional

import torch
from tt_sketch.tensor import TensorTrain


def sketch_omega_tt(
    left_sketch, right_sketch, **kwargs
):
    return left_sketch.T @ right_sketch


def sketch_psi_tt(
    left_sketch,
    right_sketch,
    *,
    tensor: TensorTrain,
    mu: int,
    **kwargs
):
    tt_core = tensor.cores[mu]
    if left_sketch is None:
        Psi = torch.einsum("ijk,kl->ijl", tt_core, right_sketch)  # type: ignore
    elif right_sketch is None:
        Psi = torch.einsum("ij,jkl->ikl", left_sketch.T, tt_core)
    else:
        Psi = torch.einsum(
            "ij,jkl,lm->ikm",
            left_sketch.T,
            tt_core,
            right_sketch
        )
    return Psi
