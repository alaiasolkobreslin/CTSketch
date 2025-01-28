import sys
from deepsoftlog.experiments.mnist_sum.sum import train as train_sum


def main(experiment_name, config_file):
    train_functions = {'mnist_sum': train_sum,}
    assert experiment_name in train_functions.keys(), f"Experiment name must be one of {tuple(train_functions.keys())}"
    return train_functions[experiment_name](config_file)

if __name__ == "__main__":
    experiment_name = 'mnist_sum'
    digit = 4
    config_file = f'deepsoftlog/experiments/{experiment_name}/config{digit}.yaml'
    # main(*sys.argv[1:])
    main(experiment_name, config_file)
