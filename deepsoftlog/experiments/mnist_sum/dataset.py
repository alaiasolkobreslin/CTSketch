import random
from pathlib import Path
from typing import Callable, Iterable

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset, random_split
import torch

from deepsoftlog.algebraic_prover.terms.expression import Expr, Constant
from deepsoftlog.algebraic_prover.terms.list_term import to_prolog_list
from deepsoftlog.data import to_prolog_image
from deepsoftlog.data.query import Query
from deepsoftlog.data.dataset import Dataset, StaticDataset
from deepsoftlog.logic.soft_term import SoftTerm

_DATA_ROOT = str(Path(__file__).parent.parent.parent.parent / "data")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

MNIST_DATA = {
        "training": MNIST(root=_DATA_ROOT, train=True, download=True, transform=transform),
        "test": MNIST(root=_DATA_ROOT, train=False, download=True, transform=transform),
        }

def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number += d
    return number

def generate_data(n):
    if n == 4:
        MNIST_DATA['training'], MNIST_DATA['validation'], _ = random_split(MNIST_DATA['training'], [20000, 4000, 36000])
        MNIST_DATA['val'], _ = random_split(MNIST_DATA['training'], [4000, 16000])
        MNIST_DATA['test'],  _ = random_split(MNIST_DATA['test'], [4000, 6000])
    else:
        r = 6000 * n % 60000
        q = 6000 * n // 60000
        r_test = 1000 * n % 10000
        q_test = 1000 * n // 10000
        sub_train_val, _ = random_split(MNIST_DATA['training'], [r, 60000 - r])
        print(len(MNIST_DATA['training']))
        print(len(sub_train_val))
        MNIST_DATA['training'] = torch.utils.data.ConcatDataset([MNIST_DATA['training']]*q + [sub_train_val])
        MNIST_DATA['training'], MNIST_DATA['validation'] = random_split(MNIST_DATA['training'], [5000*n, 1000*n])
        MNIST_DATA['val'], _ = random_split(MNIST_DATA['training'], [1000*n, 4000*n])
        sub_test, _ = random_split(MNIST_DATA['test'], [r_test, 10000-r_test])
        MNIST_DATA['test'] = torch.utils.data.ConcatDataset([MNIST_DATA['test']]*q_test + [sub_test]) 

class MnistImagesDataset(Dataset):
    def __init__(self, split_name):
        self.split_name = split_name

    def __getitem__(self, item):
        return MNIST_DATA[self.split_name][item][0]

    def __len__(self):
        return len(MNIST_DATA[self.split_name])


class MnistDataset(StaticDataset):
    def __init__(self, split_name, allowed_labels=range(10)):
        data = [entry for entry in MNIST_DATA[split_name] if entry[1] in allowed_labels]
        super().__init__(data)
        self.split_name = split_name


class MnistQueryDataset(MnistDataset):
    def __getitem__(self, item: int) -> Query:
        image, label = super().__getitem__(item)
        image_term = to_prolog_image(image)
        label_term = Constant(label)
        return Query(Expr("mnist", image_term, label_term))


def mnist_sum_dataset(n: int, split_name: str, seed=None):
    """Returns a data for sum"""
    return MNISTOperator(
        split_name=split_name,
        function_name="sum",
        operator=sum,
        size=1,
        arity=n,
        seed=seed,
    )

def chunks(xs, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

class MNISTOperator(Dataset, TorchDataset):
    def __init__(
        self,
        split_name: str,
        function_name: str,
        operator: Callable[[list[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic data for operator(img, img) style datasets.
        :param split_name: Dataset to use (training, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.split_name = split_name
        self.dataset = MNIST_DATA[self.split_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed

        indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)

        # Build list of examples (mnist indices)
        indices = tuple(chunks(indices, self.size))
        self.data_indices = tuple(chunks(indices, self.arity))

    def __getitem__(self, i: int) -> Query:
        """Generate queries"""
        mnist_indices = self.data_indices[i]
        label = str(self._get_label(i))
        # label = list(reversed(list(label)))
        # expected_term = to_prolog_list(label)
        expected_term = to_prolog_list([label])

        # Build query
        arg_terms = [to_prolog_list([self.dataset[i][0] for i in chunk], to_prolog_image)
                     for chunk in mnist_indices]

        return Query(Expr(self.function_name, *arg_terms, expected_term))

    def _get_label(self, i: int):
        mnist_indices = self.data_indices[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            digits_to_number(self.dataset[i][1] for i in chunk)
            for chunk in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data_indices)


class SumDataset(Dataset):
    def __init__(self, digits: int, n: int):
        self.digits = digits
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        # d = int(10 ** random.randrange(0, self.digits))
        a = random.randrange(0, 10)
        b = random.randrange(0, 10)

        label = list(str(a + b))
        expected_term = to_prolog_list([label])

        to_soft = lambda x: SoftTerm(Constant(x))
        a, b = list(str(a)), list(str(b))
        a_term = to_prolog_list(a, terminal=to_soft)
        b_term = to_prolog_list(b, terminal=to_soft)
        return Query(Expr("sum", a_term, b_term, expected_term))


if __name__ == "__main__":
    print(MNIST_DATA['val'][0])
