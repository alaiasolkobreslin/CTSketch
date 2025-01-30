import sys

from deepsoftlog.experiments.mnist_addition.addition import train as train_addition
from deepsoftlog.experiments.mnist_sum.sum import train as train_sum


def main(experiment_name, config_file):
    train_functions = {'mnist_addition': train_addition, "mnist_sum": train_sum}
    assert experiment_name in train_functions.keys(), f"Experiment name must be one of {tuple(train_functions.keys())}"
    return train_functions[experiment_name](config_file)


if __name__ == "__main__":
    main(*sys.argv[1:])