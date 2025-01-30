"""
code adapted from deepproblog repo
"""
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from torch.utils.data import random_split


_DATA_ROOT = Path(__file__).parent.parent.parent.parent.parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

_full_train_set = torchvision.datasets.MNIST(
        root=str(_DATA_ROOT)+"/data/", train=True, download=True, transform=transform
    )
_full_test_set = torchvision.datasets.MNIST(
        root=str(_DATA_ROOT)+"/data/", train=False, download=True, transform=transform
    )
_train_set, _val_set = random_split(_full_train_set, [50000, 10000])

def get_datasets(arity : int):
    r = (arity * 6000) % 60000
    q = (arity * 6000) // 60000

    train_length = (r // 6) * 5
    val_length = (r // 6)
    test_length = (r // 6)

    sub_train = torch.utils.data.Subset(_train_set, range(train_length))
    sub_val = torch.utils.data.Subset(_val_set, range(val_length))
    sub_test = torch.utils.data.Subset(
            torchvision.datasets.MNIST(
                root=str(_DATA_ROOT)+"/data/", train=False, download=True, 
                transform=transform), range(test_length))
    
    if q == 0:
        datasets = {
        "train": sub_train,
        "val": sub_val,
        "full_train": _full_train_set,
        "test": sub_test,
        }
        return datasets
    
    train_val = torch.utils.data.ConcatDataset([_full_train_set]*q)
    other_train, other_val = random_split(train_val, [50000 * q, 10000 * q])
    datasets = {
        "train": torch.utils.data.ConcatDataset([other_train, sub_train]),
        "val":  torch.utils.data.ConcatDataset([other_val, sub_val]),
        "full_train": torch.utils.data.ConcatDataset([train_val, _full_train_set]),
        "test": torch.utils.data.ConcatDataset([_full_test_set]*q + [sub_test]),
    }
    return datasets

def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number

def sum_n(n: int, a:int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="sum_n",
        operator=sum,
        size=n,
        arity=a,
        seed=seed,
    )


class MNISTOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[np.array, np.array, int, List[int]]:
        l1s = self.data[index]
        label, digits = self._get_label(index)
        ls = [torch.stack([self.dataset[x][0][0] for x in l1]) for l1 in l1s]
        result = ls + [label, digits]
        return tuple(result)

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = get_datasets(arity)[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        print(f'\n LENGTH OF {self.dataset_name} DATASET: {len(self.dataset)}\n')
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass
        # final length after building (training?) sum2 30,000
        # final length after building (testing?) sum2 5,000
        # final length after building (training?) sum3 20,000
        # final length after building (testing?) sum3 3,333
        print(f"FINAL LENGTH after building: {len(self.data)}")

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        digits = [[self.dataset[j][1] for j in i] for i in mnist_indices]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [ digits_to_number(di) for di in digits ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result, digits

    def __len__(self):
        return len(self.data)
