import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import signal
import functools
import os
import errno

import numpy as np
import math
from tqdm import tqdm

from src import constants
from src import util

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_reward(prediction, ground_truth):
    if prediction == ground_truth:
        reward = 1.0
    else:
        reward = 0.0
    return reward

def validate(val_loader, model, final_output, args):
    model.eval()
    loss_value = 0
    reward_value = 0
    n = 0
    eps = np.finfo(np.float32).eps.item()
    
    iter = tqdm(val_loader, total=len(val_loader))

    with torch.no_grad():
        for i, (images, target) in enumerate(iter):
            images = tuple(image.to(args.gpu_id) for image in images)
            target = target.to(args.gpu_id)

            preds = model(images)
            final_output(model,target,args, *preds) # this populates model.rewards
            rewards = np.array(model.rewards)
            rewards_mean = rewards.mean()
            reward_value += float(rewards_mean * images[0].size(0))
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)
            policy_loss = []
            for reward, log_prob in zip(rewards, model.saved_log_probs):
                policy_loss.append(-log_prob*reward)
            policy_loss = (torch.stack(policy_loss)).sum()

            n += images[0].size(0)
            loss_value += float(policy_loss.item() * images[0].size(0))
            model.rewards = []
            model.saved_log_probs = []
            torch.cuda.empty_cache()

            iter.set_description(f"[Val][{i}] AvgLoss: {loss_value/n:.4f} AvgRewards: {rewards_mean:.4f}")
    
    avg_loss = (loss_value / n)

    return avg_loss, rewards_mean
