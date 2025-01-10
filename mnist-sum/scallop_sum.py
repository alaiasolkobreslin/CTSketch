import os
import random
from typing import *
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import wandb

import scallopy
from mnist_config import mnist_sum_loader, MNISTSumNet

class MNISTSumNNet(nn.Module):
  def __init__(self, provenance, k, digit, dispatch):
    super(MNISTSumNNet, self).__init__()
    
    # MNIST Digit Recognition Network
    self.mnist_net = MNISTSumNet(digit)
    self.digit = digit

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    for i in range(1, digit + 1):
        self.scl_ctx.add_relation(f"digit_{i}", int, input_mapping=list(range(10)))
    
    if digit == 2: self.scl_ctx.add_rule("sum(a + b) = digit_1(a), digit_2(b)")
    elif digit == 3: self.scl_ctx.add_rule("sum(a + b + c) = digit_1(a), digit_2(b), digit_3(c)")
    elif digit == 4: self.scl_ctx.add_rule("sum(a + b + c + d) = digit_1(a), digit_2(b), digit_3(c), digit_4(d)")
    elif digit == 8: self.scl_ctx.add_rule("sum(a + b + c + d + e + f + g + h) = digit_1(a), digit_2(b), digit_3(c), digit_4(d), digit_5(e), digit_6(f), digit_7(g), digit_8(h)")
    elif digit == 16: self.scl_ctx.add_rule("sum(a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p) = digit_1(a), digit_2(b), digit_3(c), digit_4(d), digit_5(e), digit_6(f), digit_7(g), digit_8(h), digit_9(i), digit_10(j), digit_11(k), digit_12(l), digit_13(m), digit_14(n), digit_15(o), digit_16(p)")
    elif digit == 24: 
      self.scl_ctx.add_rule(
        "sum(n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15 + n16 + n17 + n18) \
        = digit_1(n1), digit_2(n2), digit_3(n3), digit_4(n4), digit_5(n5), digit_6(n6), digit_7(n7), digit_8(n8), digit_9(n9),  \
          digit_10(n10), digit_11(n11), digit_12(n12), digit_13(n13), digit_14(n14), digit_15(n15), digit_16(n16), digit_17(n17), digit_18(n18), digit_19(n19), \
            digit_20(n20), digit_21(n21), digit_22(n22), digit_23(n23), digit_24(n24)")
    elif digit == 32: 
      self.scl_ctx.add_rule(
        "sum(n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15 + n16 + n17 + n18 + n19 + n20 + n21 + n22 + n23 + n24 + n25 + n26 + n27 + n28 + n29 + n30 + n31 + n32) \
        = digit_1(n1), digit_2(n2), digit_3(n3), digit_4(n4), digit_5(n5), digit_6(n6), digit_7(n7), digit_8(n8), digit_9(n9),  \
          digit_10(n10), digit_11(n11), digit_12(n12), digit_13(n13), digit_14(n14), digit_15(n15), digit_16(n16), digit_17(n17), digit_18(n18), digit_19(n19), \
          digit_20(n20), digit_21(n21), digit_22(n22), digit_23(n23), digit_24(n24), digit_25(n25), digit_26(n26), digit_27(n27), digit_28(n28), digit_29(n29), \
          digit_30(n30), digit_31(n31), digit_32(n32)")
    elif digit == 64: 
      self.scl_ctx.add_rule(
        "sum(n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15 + n16 + n17 + n18 + n19 + n20 + n21 + n22 + n23 + n24 + n25 + n26 + n27 + n28 + n29 + n30 + n31 + n32 + n33 + n34 + n35 + n36 + n37 + n38 + n39 + n40 + n41 + n42 + n43 + n44 + n45 + n46 + n47 + n48 + n49 + n50 + n51 + n52 + n53 + n54 + n55 + n56 + n57 + n58 + n59 + n60 + n61 + n62 + n63 + n64) \
        = digit_1(n1), digit_2(n2), digit_3(n3), digit_4(n4), digit_5(n5), digit_6(n6), digit_7(n7), digit_8(n8), digit_9(n9),  \
          digit_10(n10), digit_11(n11), digit_12(n12), digit_13(n13), digit_14(n14), digit_15(n15), digit_16(n16), digit_17(n17), digit_18(n18), digit_19(n19), \
          digit_20(n20), digit_21(n21), digit_22(n22), digit_23(n23), digit_24(n24), digit_25(n25), digit_26(n26), digit_27(n27), digit_28(n28), digit_29(n29), \
          digit_30(n30), digit_31(n31), digit_32(n32), digit_33(n33), digit_34(n34), digit_35(n35), digit_36(n36), digit_37(n37), digit_38(n38), digit_39(n39), \
          digit_40(n40), digit_41(n41), digit_42(n42), digit_43(n43), digit_44(n44), digit_45(n45), digit_46(n46), digit_47(n47), digit_48(n48), digit_49(n49), \
          digit_50(n50), digit_51(n51), digit_52(n52), digit_53(n53), digit_54(n54), digit_55(n55), digit_56(n56), digit_57(n57), digit_58(n58), digit_59(n59), \
          digit_60(n60), digit_61(n61), digit_62(n62), digit_63(n63), digit_64(n64)")

    # The `sum_n` logical reasoning module
    self.sum_n = self.scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(digit*9 + 1)], dispatch=dispatch)

  def forward(self, x):
    # First recognize the digits
    distrs = self.mnist_net(x)
    
    # Then execute the reasoning module; the result is a size 19 tensor
    if self.digit == 2: output = self.sum_n(digit_1=distrs[0], digit_2=distrs[1])
    elif self.digit == 3: output = self.sum_n(digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2])
    elif self.digit == 4: output = self.sum_n(digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3])   
    elif self.digit == 8: output = self.sum_n(digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3], digit_5=distrs[4], digit_6=distrs[5], digit_7=distrs[6], digit_8=distrs[7])   
    elif self.digit == 16: output = self.sum_n(digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3], digit_5=distrs[4], digit_6=distrs[5], digit_7=distrs[6], digit_8=distrs[7], digit_9=distrs[8], digit_10=distrs[9], digit_11=distrs[10], digit_12=distrs[11], digit_13=distrs[12], digit_14=distrs[13], digit_15=distrs[14], digit_16=distrs[15])   
    elif self.digit == 24: 
      output = self.sum_n(
        digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3], digit_5=distrs[4], digit_6=distrs[5], digit_7=distrs[6], digit_8=distrs[7], digit_9=distrs[8], 
        digit_10=distrs[9], digit_11=distrs[10], digit_12=distrs[11], digit_13=distrs[12], digit_14=distrs[13], digit_15=distrs[14], digit_16=distrs[15], digit_17=distrs[16], digit_18=distrs[17], digit_19=distrs[18],
        digit_20=distrs[19], digit_21=distrs[20], digit_22=distrs[21], digit_23=distrs[22], digit_24=distrs[23])      
    elif self.digit == 32: 
      output = self.sum_n(
        digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3], digit_5=distrs[4], digit_6=distrs[5], digit_7=distrs[6], digit_8=distrs[7], digit_9=distrs[8], 
        digit_10=distrs[9], digit_11=distrs[10], digit_12=distrs[11], digit_13=distrs[12], digit_14=distrs[13], digit_15=distrs[14], digit_16=distrs[15], digit_17=distrs[16], digit_18=distrs[17], digit_19=distrs[18], 
        digit_20=distrs[19], digit_21=distrs[20], digit_22=distrs[21], digit_23=distrs[22], digit_24=distrs[23], digit_25=distrs[24], digit_26=distrs[25], digit_27=distrs[26], digit_28=distrs[27], digit_29=distrs[28], 
        digit_30=distrs[29], digit_31=distrs[30], digit_32=distrs[31])   
    elif self.digit == 64: 
      output = self.sum_n(
        digit_1=distrs[0], digit_2=distrs[1], digit_3=distrs[2], digit_4=distrs[3], digit_5=distrs[4], digit_6=distrs[5], digit_7=distrs[6], digit_8=distrs[7], digit_9=distrs[8], 
        digit_10=distrs[9], digit_11=distrs[10], digit_12=distrs[11], digit_13=distrs[12], digit_14=distrs[13], digit_15=distrs[14], digit_16=distrs[15], digit_17=distrs[16], digit_18=distrs[17], digit_19=distrs[18], 
        digit_20=distrs[19], digit_21=distrs[20], digit_22=distrs[21], digit_23=distrs[22], digit_24=distrs[23], digit_25=distrs[24], digit_26=distrs[25], digit_27=distrs[26], digit_28=distrs[27], digit_29=distrs[28], 
        digit_30=distrs[29], digit_31=distrs[30], digit_32=distrs[31], digit_33=distrs[32], digit_34=distrs[33], digit_35=distrs[34], digit_36=distrs[35], digit_37=distrs[36], digit_38=distrs[37], digit_39=distrs[38], 
        digit_40=distrs[39], digit_41=distrs[40], digit_42=distrs[41], digit_43=distrs[42], digit_44=distrs[43], digit_45=distrs[44], digit_46=distrs[45], digit_47=distrs[46], digit_48=distrs[47], digit_49=distrs[48], 
        digit_50=distrs[49], digit_51=distrs[50], digit_52=distrs[51], digit_53=distrs[52], digit_54=distrs[53], digit_55=distrs[54], digit_56=distrs[55], digit_57=distrs[56], digit_58=distrs[57], digit_59=distrs[58], 
        digit_60=distrs[59], digit_61=distrs[60], digit_62=distrs[61], digit_63=distrs[62], digit_64=distrs[63])   

    return output

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
  return F.binary_cross_entropy(output, gt)

def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance, digits, dispatch, save_model = False):
    self.model_dir = model_dir
    self.network = MNISTSumNNet(provenance, k, digits, dispatch).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.best_loss = None
    self.save_model = save_model
    self.digits = digits

    if loss == "nll": self.loss = nll_loss
    elif loss == "bce": self.loss = bce_loss
    else: raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    train_loss = 0
    for (data, target, _) in iter:
      self.optimizer.zero_grad()
      data = tuple([data_i.to(device) for data_i in data])
      target = target.to(device)
      output = self.network(data)
      loss = self.loss(output, target)
      train_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, _) in iter:
        data = tuple([data_i.to(device) for data_i in data])
        target = target.to(device)        
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = float(correct/num_items)
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc * 100:.2f}%)")
      
      if self.save_model and self.best_acc < perc:
        self.best_loss = test_loss
        self.best_acc = perc
        torch.save(self.network, os.path.join(model_dir, f"sum{digits}_best.pt"))
    return perc, test_loss

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_loss = self.test_epoch(epoch)
      t2 = time()

      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--digits", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=0.0005)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  digits = args.digits
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance

  if torch.cuda.is_available(): device = torch.device(1)
  else: device = torch.device('cpu')

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scallop_sum"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digits)
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance, digits, args.dispatch)
  print(len(train_loader.dataset))
  print(len(test_loader.dataset))

  # setup wandb
  config = vars(args)
  wandb.init(
    project=f"scallop_sum",
    name = f"{digits}_{args.seed}",
    config=config)
  print(config)
  
  trainer.train(n_epochs)