import os
import random
from typing import *
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb
import pickle

from scene_configs import scene_loader, SceneNet, prepare_inputs, scenes, objects

queries = {}
with open('scene/llm_single.pkl', 'rb') as f: 
  queries = pickle.load(f)

def parse_response(answer):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  for s in random_scene:
    if s in answer: return s
  raise Exception("LLM failed to provide an answer") 

def objects_scene():
  gt = torch.zeros((len(objects), len(scenes)))
  for i, o in enumerate(objects):
    if o == 'skip' or o == 'ball': 
      continue
    s = parse_response(queries[o])
    gt[i, scenes.index(s)] = 1
  return gt

def gt_program(data, target, lens):
    pred = []
    dim = data.shape[:-1].numel()
    data = data.flatten(0,-2)
    for d in range(dim):
      ind = 0
      for n in range(len(lens)):
        i = data[d][ind:ind+lens[n]]
        ind += lens[n]
        input = [objects[int(j)] for j in i]
        input.sort()
    acc = torch.where(torch.stack(pred).to(device).reshape(dim, -1) == target, 1., 0.)
    return acc

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, seed, save_model=False):
    self.model_dir = model_dir
    self.network = SceneNet().to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.seed = seed
    self.dict = {}
    self.save_model = save_model
    
    with open('scene/yolo_only.pkl', 'rb') as f: 
      self.dict = pickle.load(f)

  def pretrain(self):
      t0 = time()
      self.gt = objects_scene()
      t1 = time()
      wandb.log({"pretrain": t1 - t0})
      self.gt = self.gt.to(device)

  def program(self, inputs, lens):
    output = torch.einsum('ab, bc -> ac', inputs, self.gt)
    output = [output[lens[:i].sum():lens[:i+1].sum()].mean(dim=0) for i in range(len(lens))]
    output = torch.stack(output, dim=0)
    return output

  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
    return F.cross_entropy(output, gt)

  def train_epoch(self, epoch):
    train_loss = 0
    self.network.train()
    for (input, file, target) in self.train_loader:
      self.optimizer.zero_grad()
      box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
      target = target.to(device)
      output_t = self.network(input.to(device), cls.to(device), conf.to(device))
      output = self.program(output_t, box_len)
      output = F.normalize(output, dim=-1)
      loss = self.loss_fn(output, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    print(f"[Epoch {epoch}] {train_loss}")
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    with torch.no_grad():
      for (input, file, target) in self.test_loader:
        box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
        target = target.to(device)
        output_t = self.network(input.to(device), cls.to(device), conf.to(device))
        output = self.program(output_t, box_len)
        pred = output.argmax(dim=-1)
        correct += (pred == target).sum()   
    perc = float(correct/num_items)
    print(f"[Test] {int(correct)}/{int(num_items)} ({perc * 100:.2f})%")    
    
    if self.save_model and self.best_acc < perc:
      self.best_acc = perc
      torch.save(self.network, model_dir+f"/{self.seed}_best.pkl")

    return perc

  def train(self, n_epochs):
    self.pretrain()
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      acc = self.test()
      t2 = time()

      wandb.log({
        "train_loss": float(train_loss),
        "test_acc": float(acc),
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("scene")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    
    # Parameters
    seed = args.seed

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    torch.manual_seed(seed)
    random.seed(seed)

    # Data
    data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/scene"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene/"))
    os.makedirs(model_dir, exist_ok=True)

    (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
    trainer = Trainer(train_loader, test_loader, model_dir, args.learning_rate, seed)

    # Setup wandb
    config = vars(args)
    run = wandb.init(
      project=f"tensor_scene",
      config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)
    run.finish()