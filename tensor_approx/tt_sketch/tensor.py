"""Implements various types of tensors"""
from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Iterable,
    List,
    Optional,
    Sequence,
    Dict,
    Tuple,
    Union,
    TypeVar,
    Generic,
)
import warnings

import torch
from numpy.random import SeedSequence

from tt_sketch.utils import ArrayList, TTRank, process_tt_rank, random_normal

TType = TypeVar("TType", bound="Tensor")


class Tensor(ABC):
    """Abstract base class for tensors."""

    #: The shape of the tensor
    shape: Tuple[int, ...]

    @abstractproperty
    def T(self: TType) -> TType:
        """Transpose of the tensor.

        If a tensor has shape (n1,n2,...,nd), then transpose of the tensor
        has shape (nd,...,n2,n1).
        """

    @abstractproperty
    def size(self) -> int:
        """Number of floating point elements used to store the tensor."""

    @abstractmethod
    def to_numpy(self):
        """Converts the tensor to a (dense) numpy array of same shape."""

    def error(
        self: TType,
        other: TType,
        relative: bool = False,
        rmse: bool = False,
        fast: bool = False,
    ) -> float:
        """L2 error of the tensor.

        If ``fast=True``, then error is computed using inner product formula.
        This is not numerically stable, and gives inaccurate results below
        relative errors of around 1e-8.
        """
        if isinstance(other, torch.Tensor):
            other = DenseTensor(other)
        other_norm = other.norm()
        if fast:
            self_norm = self.norm()
            dot = self.dot(other)
            norm_sum = self_norm**2 + other_norm**2
            error = torch.sqrt(norm_sum) * torch.sqrt(torch.abs(1 - 2 * dot / norm_sum))
        else:
            error = torch.linalg.norm(self.to_numpy() - other.to_numpy())
        if relative:
            if other_norm == 0:
                return torch.inf
            error /= other_norm
        if rmse:
            error /= torch.sqrt(torch.prod(self.shape))
        return error

    @property
    def ndim(self) -> int:
        """Number of modes of the tensor."""
        return len(self.shape)

    def dense(self) -> DenseTensor:
        """Converts to ``DenseTensor`` object"""
        dense_tensor = self.to_numpy()
        return DenseTensor(dense_tensor)

    def __add__(self, other) -> TensorSum:
        """Addition of two tensors produces ``TensorSum`` object"""
        if isinstance(other, TensorSum):
            if not isinstance(self, TensorSum):
                return TensorSum([self] + other.tensors)
            else:
                return TensorSum(self.tensors + other.tensors)
        elif isinstance(self, TensorSum):
            return TensorSum(self.tensors + [other])
        else:
            return TensorSum([self, other])

    @abstractmethod
    def __mul__(self: TType, other: float) -> TType:
        """Multiplication by a scalar"""

    def __rmul__(self: TType, other: float) -> TType:
        return self.__mul__(other)

    def __truediv__(self, other: float):
        return self.__mul__(1 / other)

    def __sub__(self, other: Tensor) -> Tensor:
        return self + (-other)

    def __neg__(self) -> Tensor:
        return self * -1

    def dot(self: TType, other: TType, reverse=False) -> float:
        """Dot product of two tensors"""
        if isinstance(other, TensorSum):
            return other.dot(self)
        if not reverse:  # try first to see if other can dot self
            return other.dot(self, reverse=True)
        self_np = self.to_numpy().reshape(-1)
        other_np = other.to_numpy().reshape(-1)
        return torch.matmul(self_np, other_np)

    def norm(self) -> float:
        """L2 norm of the tensor"""
        # torch.abs because dot can be negative due to numerical errors
        return torch.sqrt(torch.abs(self.dot(self)))

    def __matmul__(self: TType, other: TType) -> float:
        return self.dot(other)


class DenseTensor(Tensor):
    shape: Tuple[int, ...]
    data: torch.Tensor

    def __init__(self, data) -> None:
        self.shape = data.shape
        self.data = data

    def to_sparse(self) -> SparseTensor:
        """
        Converts to a sparse tensor.

        This is mainly used for testing sketching algorithms, as there is
        otherwise no reason to convert a dense tensor to a sparse tensor.
        """
        X = self.data
        n_dims = len(X.shape)
        indices = torch.meshgrid([torch.arange(s, dtype=torch.int64) for s in X.shape], indexing='ij')
        inds = torch.stack(indices).reshape(n_dims, -1)
        entries = X.reshape(-1)
        return SparseTensor(X.shape, inds, entries)

    @property
    def T(self) -> DenseTensor:
        permutation = tuple(range(len(self.shape))[::-1])
        new_data = self.data.permute(permutation)
        return self.__class__(new_data)

    @property
    def size(self) -> int:
        return torch.prod(self.shape)

    def to_numpy(self):
        return self.data

    def __repr__(self) -> str:
        return f"<Dense tensor of shape {self.shape} at {hex(id(self))}>"

    def __mul__(self, other: float) -> DenseTensor:
        return self.__class__(self.data * other)

    @classmethod
    def random(cls, shape: Tuple[int, ...]) -> DenseTensor:
        return cls(random_normal(shape))


@dataclass
class SparseTensor(Tensor):
    #: The shape of the tensor
    shape: Tuple[int, ...]

    #: The indices of the non-zero entries
    indices: Union[torch.Tensor, Tuple[torch.Tensor, ...]]

    #: The entries of the non-zero entries
    entries: torch.Tensor

    def __post_init__(self):
        if isinstance(self.indices, tuple):
            self.indices = torch.stack(self.indices)

    @property
    def T(self) -> SparseTensor:
        new_indices = torch.flip(self.indices, (0,))
        new_shape = torch.flip(torch.tensor(self.shape), (0,))
        return self.__class__(new_shape, new_indices, self.entries)

    @property
    def size(self) -> int:
        return self.nnz * (self.ndim + 1)

    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return len(self.entries)

    def split(self, n_summands: int) -> TensorSum:
        """
        Splits the tensor a ``TensorSum`` containing ``n_summands`` tensors.
        """
        block_size = self.nnz // n_summands
        summands: List[Tensor] = []
        for i in range(n_summands):
            if i < n_summands - 1:
                summand_slice = slice(i * block_size, (i + 1) * block_size)
            else:
                summand_slice = slice(i * block_size, self.nnz)
            summand_indices = tuple(ind[summand_slice] for ind in self.indices)
            summands.append(
                SparseTensor(
                    self.shape,
                    summand_indices,
                    self.entries[summand_slice],
                )
            )
        return TensorSum(summands)

    def to_numpy(self):
        X = torch.zeros(self.shape)
        X[tuple(self.indices)] = self.entries
        return X

    def norm(self) -> float:
        return float(torch.linalg.norm(self.entries))

    def __repr__(self) -> str:
        return (
            f"<Sparse tensor of shape {self.shape} with {self.nnz} non-zero"
            f" entries at {hex(id(self))}>"
        )

    def dot(self, other: Tensor, reverse=False) -> float:
        try:
            other_entries = other.gather(self.indices)
            return other_entries.matmul(self.entries)
        except AttributeError:
            return super().dot(other, reverse=reverse)

    @classmethod
    def random(
        cls, shape: Tuple[int, ...], nnz: int, seed: Optional[int] = None
    ) -> SparseTensor:
        """
        Generates a random sparse tensor with `nnz` non-zero gaussian entries
        """
        if seed is not None:
            torch.manual_seed(seed)
        total_size = torch.prod(shape)
        indices_flat = torch.randint(total_size, (nnz,))
        indices = torch.div(indices_flat, shape[1], rounding_mode='floor'), indices_flat % shape[1]
        entries = random_normal(shape=(nnz,), seed=seed)
        return cls(shape, indices, entries)

    def __mul__(self, other: float) -> SparseTensor:
        return self.__class__(self.shape, self.indices, self.entries * other)

    def gather(
        self, indices
    ):
        """
        Gathers the entries corresponding to the given indices.
        """
        
        dense_indices = indices[0] * self.shape[1] + indices[1]
        out = torch.zeros(len(dense_indices))
        dic = self.dict
        for i, ind in enumerate(dense_indices):
            out[i] = dic.get(ind, 0.0)
        return out

    @cached_property
    def dict(self) -> Dict[int, float]:
        dense_indices = self.indices[0] * self.shape[1] + self.indices[1]
        return {i: v for i, v in zip(dense_indices, self.entries)}


class TensorTrain(Tensor):

    #: The shape of the tensor
    shape: Tuple[int, ...]

    #: Tuple encoding the tensor train rank
    rank: Tuple[int, ...]

    #: A list containing the cores of the tensor train
    cores: ArrayList

    def __init__(self, cores: ArrayList) -> None:
        self.cores = cores
        self.shape = tuple(C.shape[1] for C in cores)
        self.rank = tuple(C.shape[0] for C in cores[1:])

    @property
    def T(self) -> TensorTrain:
        new_cores = [C.permute(2, 1, 0) for C in self.cores[::-1]]
        return self.__class__(new_cores)

    def to_numpy(self):
        dense_tensor = self.cores[0]
        dense_tensor = dense_tensor.reshape(dense_tensor.shape[1:])
        for C in self.cores[1:]:
            dense_tensor = torch.einsum("...j,jkl->...kl", dense_tensor, C)
        dense_tensor = dense_tensor.reshape(dense_tensor.shape[:-1])
        return dense_tensor

    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        rank: TTRank,
        seed: Optional[int] = None,
        orthog: bool = False,
        trim: Optional[bool] = None,
        norm_goal: str = "norm-1",
    ) -> TensorTrain:
        """
        Generate random orthogonal tensor train cores

        By default, a core of ``(r1, n, r2)`` has Gaussian entries with zero
        mean and variance ``1 / r1 * n * r2``, so that expected Frobenius norm
        is 1.

        If ``trim=True``, the ranks are trimmed; i.e. we enforce that r1*n>=r2
        and r2*n>=r1a.

        If ``orthog`` is set to ``True``, all cores except the last are
        left-orthogonalized. Trim must be enabled in this case.
        """

        d = len(shape)
        if trim is None:
            trim = True if orthog else False
        if orthog and not trim:
            raise ValueError(
                "Trimming must be enabled if orthogonalization is enabled."
            )
        rank = process_tt_rank(rank, shape, trim=trim)
        rank_augmented = (1,) + tuple(rank) + (1,)

        cores = []
        seq = SeedSequence(seed)
        seeds = seq.generate_state(d)
        # seeds = torch.randint(0, 10000, (d,))
        for i in range(d):
            r1 = rank_augmented[i]
            r2 = rank_augmented[i + 1]
            n = shape[i]
            core = random_normal(shape=(r1 * n, r2), seed=seeds[i])

            if orthog and i < d - 1:
                core, _ = torch.linalg.qr(core, mode="reduced")
            elif norm_goal == "norm-1":
                core /= torch.sqrt(torch.tensor(r1 * n))
            elif norm_goal == "norm-preserve":
                core /= torch.sqrt(torch.tensor(r1))
            else:
                raise ValueError(f"Unknown norm goal: {norm_goal}")

            core = core.reshape(r1, n, r2)
            cores.append(core)

        return cls(cores)

    @classmethod
    def zero(cls, shape: Tuple[int, ...], rank: TTRank) -> TensorTrain:
        d = len(shape)

        rank = process_tt_rank(rank, shape, trim=False)
        cores = []
        for (r1, d, r2) in zip((1,) + rank, shape, rank + (1,)):
            cores.append(torch.zeros((r1, d, r2)))
        return cls(cores)

    def partial_dense(self, dir: str = "lr"):
        """Do partial contractions to dense tensor; ``X[0].X[1]...X[mu]``"""
        if dir == "lr":
            partial_cores = [self.cores[0].reshape(-1, self.cores[0].shape[-1])]
            for core in self.cores[1:-1]:
                new_core = torch.einsum("ij,jkl->ikl", partial_cores[-1], core)
                new_core = new_core.reshape(-1, new_core.shape[-1])
                partial_cores.append(new_core)
        elif dir == "rl":
            partial_cores = [
                self.cores[-1].reshape(self.cores[-1].shape[0], -1)
            ]
            for core in self.cores[-2:0:-1]:
                new_core = torch.einsum("ijk,kl->ijl", core, partial_cores[-1])
                new_core = new_core.reshape(new_core.shape[0], -1)
                partial_cores.append(new_core)
        return partial_cores

    def __getitem__(self, index: int):
        return self.cores[index]

    def __setitem__(self, index: int, data) -> None:
        self.cores[index] = data

    def gather(
        self,
        idx,
    ):
        """Gather entries of dense tensor according to indices.

        For each row of `idx` this returns one number. This number is obtained
        by multiplying the slices of each core corresponding to each index (in
        a left-to-right fashion).
        """
        if not isinstance(idx, torch.Tensor):
            idx_array = torch.stack(idx)
        else:
            idx_array = torch.tensor(idx)
        N = idx_array.shape[1]
        result = torch.index_select(self[0].reshape(self[0].shape[1:]), 0, idx_array[0])
        for i in range(1, self.ndim):
            r = self[i].shape[2]
            next_step = torch.zeros((N, r))
            for j in range(self.shape[i]):
                idx_mask = torch.where(idx_array[i] == j)
                mat = self[i][:, j, :]
                next_step[idx_mask] = result[idx_mask] @ mat
            result = next_step
        return result.reshape(-1)

    def norm(self) -> float:
        self_orth = self.orthogonalize()
        return torch.linalg.norm(self_orth.cores[-1])

    def round(
        self,
        eps: Optional[float] = None,
        max_rank: Optional[TTRank] = None,
        orthogonalized: bool = False,
    ) -> TensorTrain:
        """Standard TT-SVD rounding scheme.

        First left orthogonalize in LR sweep, then apply SVD-based rounding in a
        RL sweep. Leaves the tensor right-orthogonalized.

        If the tensor is already orthogonalized, then pass
        ``orthogonalized=True`` to avoid unnecessary re-orthogonalization."""
        if not orthogonalized:
            tt = self.orthogonalize()  # left-orthogonalize
        else:
            tt = self
        if eps is None:
            eps = 0
        if max_rank is None:
            max_rank = tt.rank
        max_rank = process_tt_rank(max_rank, tt.shape, trim=True)

        new_cores = []
        US_trunc
        for mu, C in list(enumerate(tt.cores))[::-1]:
            if mu < tt.ndim - 1:
                C = torch.einsum("ijk,kl->ijl", C, US_trunc)
            if mu > 0:
                C_mat = C.reshape(C.shape[0], C.shape[1] * C.shape[2])
                U, S, Vt = torch.linalg.svd(C_mat)
                r = max(1, min(torch.sum(S > S[0] * eps), max_rank[mu - 1]))
                US_trunc = U[:, :r] @ torch.diag(S[:r])
                Vt_trunc = Vt[:r, :]
                new_cores.append(Vt_trunc.reshape(r, C.shape[1], C.shape[2]))
            else:
                new_cores.append(C)

        return self.__class__(new_cores[::-1])

    def svdvals(self):
        """Return singular value of each mode"""
        tt = self.orthogonalize()
        svdvals = []
        U
        S
        for mu, C in list(enumerate(tt.cores))[::-1]:
            if mu < tt.ndim - 1:
                US = U @ torch.diag(S)
                C = torch.einsum("ijk,kl->ijl", C, US)

            if mu > 0:
                C_mat = C.reshape(C.shape[0], C.shape[1] * C.shape[2])
            else:
                C_mat = C.reshape(C.shape[0] * C.shape[1], C.shape[2])

            U, S, _ = torch.linalg.svd(C_mat)
            svdvals.append(S)
        svdvals = svdvals[::-1]
        return svdvals

    def __mul__(self, other: float) -> TensorTrain:
        new_cores = deepcopy(self.cores)
        # Absorb number into last core, since if we do orthogonalize, we left
        # orthogonalize.
        new_cores[-1] = new_cores[-1] * other
        return self.__class__(new_cores)

    __rmul__ = __mul__

    @property
    def size(self) -> int:
        return sum(core.size for core in self.cores)

    def add(self, other: TensorTrain) -> TensorTrain:
        """
        Add two ``TensorTrain`` by taking direct sums of cores.

        Note, this does not overload the addition operator ``+``; the addition
        operator instead returns a lazy ``TensorSum`` object.
        """
        new_cores = [torch.concat((self.cores[0], other.cores[0]), axis=2)]
        for C1, C2 in zip(self.cores[1:-1], other.cores[1:-1]):
            r1, d, r2 = C1.shape
            r3, _, r4 = C2.shape
            zeros1 = torch.zeros((r1, d, r4))
            zeros2 = torch.zeros((r3, d, r2))
            row1 = torch.concat((C1, zeros1), axis=2)
            row2 = torch.concat((zeros2, C2), axis=2)
            new_cores.append(torch.concat((row1, row2), axis=0))
        new_cores.append(
            torch.concat((self.cores[-1], other.cores[-1]), axis=0)
        )
        new_tt = self.__class__(new_cores)
        return new_tt

    def dot(self, other: Tensor, reverse=False) -> float:
        """
        Compute the dot product of two tensor trains with the same shape.

        Result is computed in a left-to-right sweep.
        """
        if isinstance(other, TensorTrain):
            result = torch.einsum("ijk,ljm->km", self.cores[0], other.cores[0])
            for core1, core2 in zip(self.cores[1:], other.cores[1:]):
                # optimize reduces complexity from r^4*n to r^3*n
                result = torch.einsum("ij,ika,jkb->ab", result, core1, core2)
            return torch.sum(result)
        else:
            return super().dot(other, reverse=reverse)

    def orthogonalize(self) -> TensorTrain:
        """Do QR sweep to left-orthogonalize"""
        new_cores = []
        R
        for mu, C in enumerate(self.cores):
            if mu > 0:
                C = torch.einsum("ij,jkl->ikl", R, C)
            if mu < self.ndim - 1:
                C_mat = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
                Q, R = torch.linalg.qr(C_mat)
                new_cores.append(Q.reshape(C.shape[0], C.shape[1], -1))
            else:
                new_cores.append(C)
        return self.__class__(new_cores)

    def __repr__(self) -> str:
        return (
            f"<Tensor train of shape {self.shape} with rank {self.rank}"
            f" at {hex(id(self))}>"
        )

    def error(
        self: TType,
        other: TType,
        relative: bool = False,
        rmse: bool = False,
        fast: bool = False,
    ) -> float:
        """
        L2 error of tensor, see docs for ``Tensor::error``.

        Overloads only the error between two Tensor Trains
        using much faster accurate method.
        """
        try:  # Try coercing to TensorTrain
            other = other.to_tt()
        except AttributeError:
            pass

        if isinstance(other, TensorTrain):
            error = self.add(-other).norm()
            if relative:
                other_norm = other.norm()
                if other_norm == 0:
                    return torch.inf
                error /= other_norm
            if rmse:
                error /= torch.sqrt(torch.prod(self.shape))
            return error
        else:
            return super().error(other, relative=relative, rmse=rmse, fast=fast)


class TensorSum(Generic[TType], Tensor):
    """Container for sums of tensors"""

    shape: Tuple[int, ...]
    tensors: List[TType]

    def __init__(self, tensors: List[TType], shape=None) -> None:
        if shape is None:
            shape = tensors[0].shape
        self.shape = shape
        self.tensors = tensors

    @property
    def size(self) -> int:
        return sum(tensor.size for tensor in self.tensors)

    @property
    def T(self) -> TensorSum:
        new_tensors: List[TType] = [X.T for X in self.tensors]
        return self.__class__(new_tensors)

    def to_numpy(self):
        s = torch.zeros(self.shape)
        for X in self.tensors:
            s += X.to_numpy()
        return s

    def __add__(self, other: TType) -> TensorSum:
        if isinstance(other, TensorSum):
            return self.__class__(self.tensors + other.tensors)
        else:
            return self.__class__(self.tensors + [other])

    def __iadd__(self, other: TType) -> TensorSum:
        if isinstance(other, TensorSum):
            self.tensors.extend(other.tensors)
        else:
            self.tensors.append(other)
        return self

    def __repr__(self) -> str:
        return (
            f"<Sum of {self.num_summands} tensors of shape {self.shape}"
            f" at {hex(id(self))}>"
        )

    @property
    def num_summands(self) -> int:
        return len(self.tensors)

    def __mul__(self, other: Union[float, Iterable[float]]) -> TensorSum:
        try:
            return self.__class__(
                [X * c for X, c in zip(self.tensors, other, strict=True)]
            )
        except TypeError:
            return self.__class__([X * other for X in self.tensors])

    def dot(self, other: Tensor, reverse=False) -> float:
        return sum(X.dot(other, reverse) for X in self.tensors)


class CPTensor(Tensor):
    """Implements CP tensors.

    The cores are stored as list of shape ``(shape[i], rank)``."""

    shape: Tuple[int, ...]
    rank: int
    cores: ArrayList

    def __init__(self, cores: ArrayList) -> None:
        self.cores = cores
        self.rank = cores[0].shape[1]
        self.shape = tuple(C.shape[0] for C in cores)

    def size(self) -> int:
        return sum(C.size for C in self.cores)

    @property
    def T(self) -> CPTensor:
        new_cores = self.cores[::-1]
        return self.__class__(new_cores)

    def to_numpy(self):
        dense_tensor = self.cores[0]  # shape (n0,r)
        for C in self.cores[1:]:
            dense_tensor = torch.einsum(
                "...j,ij->...ij", dense_tensor, C
            )  # shape (n0,...,nk,r)
        dense_tensor = torch.sum(dense_tensor, axis=-1)
        return dense_tensor

    @classmethod
    def random(
        cls, shape: Tuple[int, ...], rank: int, seed: Optional[int] = None
    ) -> CPTensor:
        d = len(shape)
        seq = SeedSequence(seed)
        seeds = seq.generate_state(d)
        cores = []
        for i in range(d):
            core = random_normal(shape=(shape[i], rank), seed=seeds[i])
            core /= torch.sqrt(shape[i])
            cores.append(core)

        return cls(cores)

    def __getitem__(self, index: int):
        return self.cores[index]

    def __setitem__(self, index: int, data) -> None:
        self.cores[index] = data

    def gather(self, idx):
        """Obtain the values of the tensor at the given indices."""

        res = 1
        for C, id in zip(self.cores, idx):
            res *= C[id]
        return torch.sum(res, axis=1)

    def __repr__(self) -> str:
        return (
            f"<CP tensor of shape {self.shape} and rank {self.rank} "
            f"at {hex(id(self))}>"
        )

    def __mul__(self, other: float) -> CPTensor:
        new_cores = [c for c in self.cores]
        new_cores[0] = new_cores[0] * other
        return self.__class__(new_cores)


class TuckerTensor(Tensor):
    """Implements Tucker tensors.

    This consists of a core tensor of shape ``(s1, ..., sd)`` and ``d``
    factor matrices of shape ``(si, ni)``, where ``(n1, ..., nd)`` is the
    overall shape of the tensor."""

    def __init__(self, factors: ArrayList, core) -> None:
        self.core = core
        self.factors = factors

        self.shape = tuple(U.shape[1] for U in factors)
        self.rank = tuple(U.shape[0] for U in factors)

    @property
    def T(self) -> TuckerTensor:
        new_factors = self.factors[::-1]
        permutation = tuple(range(len(self.shape))[::-1])
        new_core = self.core.permute(permutation)
        return self.__class__(new_factors, new_core)

    @property
    def size(self) -> int:
        return self.core.size + sum(U.size for U in self.factors)

    def to_numpy(self) :
        core_contracted = self.core
        for i, U in enumerate(self.factors):
            left_dim = torch.prod(self.shape[:i]).long()
            right_dim = torch.prod(self.rank[i + 1 :]).long()
            core_mat = core_contracted.reshape(
                left_dim, self.rank[i], right_dim
            )
            core_contracted = torch.einsum("ijk,jl->ilk", core_mat, U)
        return core_contracted.reshape(self.shape)

    def __mul__(self, other: float) -> TuckerTensor:
        new_core = self.core * other
        return self.__class__(self.factors, new_core)

    def __repr__(self) -> str:
        return (
            f"<Tucker tensor of shape {self.shape} and rank {self.rank} "
            f"at {hex(id(self))}>"
        )

    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        rank: Union[int, Tuple[int, ...]],
        seed: Optional[int] = None,
    ) -> TuckerTensor:
        d = len(shape)
        try:
            rank_tuple = tuple(rank)  # type: ignore
        except TypeError:
            rank_tuple = (rank,) * d  # type: ignore
        rank_tuple = tuple(min(r1, r2) for r1, r2 in zip(rank_tuple, shape))

        seq = SeedSequence(seed)
        core_seed = seq.generate_state(1)[0]
        # master_generator = torch.Generator().manual_seed(seed)
        # core_seed = int(master_generator.initial_seed())
        core = random_normal(shape=rank_tuple, seed=core_seed)
        factors = []
        seeds = seq.generate_state(d)
        for r, n, seed in zip(rank_tuple, shape, seeds):
            U = random_normal(shape=(r, n), seed=seed)
            U = torch.linalg.qr(U.T)[0].T
            factors.append(U)

        return cls(factors, core)
