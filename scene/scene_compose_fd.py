import torch
import torch.nn.functional as F
from torch import nn

import csv
from argparse import ArgumentParser
import os
import random

from dataset import scene_loader, scenes, SceneNet, objects, Trainer, objects_long
from torch_modules.finite_difference import FiniteDifference, ListInputMapping, DiscreteInputMapping, ListOutputMapping
from scene.configs import classify_compose

class ISEDLinearNet(nn.Module):
  def __init__(self):
    super(ISEDLinearNet, self).__init__()
    # self.linear1 = nn.Linear(9, 9)
    self.linear2 = nn.Linear(9, 9)

  def forward(self, x):
    # x = F.relu(self.linear1(x))
    x = F.softmax(self.linear2(x), dim=1)
    return x

class ISEDSceneNet(nn.Module):
    def __init__(self):
        super(ISEDSceneNet, self).__init__()
        self.net = SceneNet()
        self.max_det = 10
        self.linear_net = ISEDLinearNet()
    
    def forward(self, x, pred, box_len, conf):
      x = self.net(x, pred, conf)
      bbox = FiniteDifference(**{
        "bbox": classify_compose,
        "input_mappings": (ListInputMapping(box_len, self.max_det, DiscreteInputMapping(objects)),),
        "output_mapping": ListOutputMapping(scenes),
        })
      x = [F.pad(x[box_len[:i].sum():box_len[:i+1].sum()], (0, 0, 0, self.max_det-box_len[i])) for i in range(len(box_len))]
      x = torch.stack(x, dim=0).flatten(1)
      _, probs = bbox(x)
      x = self.linear_net(probs)
      return x

if __name__ == "__main__":
  parser = ArgumentParser("scene")
  parser.add_argument("--model-name", type=str, default="scene.pkl")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=5e-3)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()


  accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
  times = ["T " + str(i+1) for i in range(args.n_epochs)]
  losses = ["L " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'grad_type'] + accuracies + times + losses

  for seed in [1234]:
    torch.manual_seed(seed)
    random.seed(seed)

    data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/scene"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene/ised"))
    if not os.path.exists(model_dir): os.makedirs(model_dir)
            
    (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
    trainer = Trainer(ISEDSceneNet(),train_loader, test_loader, args.learning_rate, model_dir, seed)

    trainer.train(args.n_epochs)