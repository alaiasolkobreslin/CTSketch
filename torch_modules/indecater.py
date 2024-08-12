import torch

import torch_modules
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import signal
import functools
import os
import errno

from src import constants
from src import util

class IndeCateR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.f = kwargs['bbox']
        self.k = kwargs['n_samples']
        self.input_mappings = kwargs['input_mappings']
        self.fn_cache = {}
        self.caching = True
        self.timeout_seconds = 1
        self.error_message = os.strerror(errno.ETIME)
        self.dim = len(self.input_mappings)
        
    def timeout_decorator(self, func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(self.error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(self.timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    
    def loss_fn(self, data, target, task):
        pred = []
        x = data.flatten(0,-2)
        for i in x:
            pred.append(task(*tuple(i)))
        acc = torch.where(torch.stack(pred).reshape(data.shape[:-1]) == target, 1., 0.)
        return acc
    
    def indecater_multiplier(self, batch_size):
        input_mapping_len = len(self.input_mappings[0].elements)
        icr_mult = torch.zeros((self.dim, input_mapping_len, self.k, batch_size, self.dim))
        icr_replacement = torch.zeros((self.dim, input_mapping_len, self.k, batch_size, self.dim))
        for i in range(self.dim):
            for j in range(10):
                icr_mult[i,j,:,:,i] = 1
                icr_replacement[i,j,:,:,i] = j
        return icr_mult, icr_replacement

    def forward(self, gt, *inputs):
        
        batch_size = inputs[0].tensor.size(0)
                
        # Prepare the inputs to the black-box function
        to_compute_inputs, sampled_indices = [], []
        distrs = []
        for (input_i, input_mapping_i) in zip(inputs, self.input_mappings):
            distr, sampled_indices_i, sampled_elements_i = input_mapping_i.sample(
                input_i, sample_count=self.k)
            to_compute = sampled_elements_i
            to_compute_inputs.append(to_compute)
            sampled_indices.append(sampled_indices_i)
            distrs.append(distr)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)
        
        results = torch.tensor(self.invoke_function_on_batched_inputs(to_compute_inputs))
        to_compare = gt.unsqueeze(-1).repeat(1, self.k)
        f_sample = torch.where(results == to_compare, 1., 0.)
        
        sampled_indices_t = [sampled_indices_i.t() for sampled_indices_i in sampled_indices]
        log_p_sample = []
        for i in range(len(distrs)):
            log_p_sample.append(torch.stack([distrs[i].log_prob(sampled_indices_t[i][j]) for j in range(self.k)]).t())
        f_mean = f_sample.mean(dim=1)
        
        # Outer samples for IndeCateR
        outer_samples = torch.stack(sampled_indices).unsqueeze(1).repeat(1, 10, 1, 1)
        outer_samples_new = torch.stack([outer_samples] * len(inputs))
        outer_samples = outer_samples_new.permute(0, 2, 4, 3, 1)
        m, r = self.indecater_multiplier(batch_size)
        outer_samples = outer_samples * (1 - m) + r
        outer_loss = self.loss_fn(outer_samples, gt.unsqueeze(0).unsqueeze(0).unsqueeze(0), self.f)
        
        variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
        logits = torch.stack([input.tensor for input in inputs])
        logits = logits.transpose(0, 1)
        indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
        indecater_expression = indecater_expression.sum(dim=-1)
        indecater_expression = indecater_expression.sum(dim=-1)

        icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
        loss = -torch.log(indecater_expression + 1e-8)
        loss = loss.mean(dim=0)
        return loss

    def zip_batched_inputs(self, batched_inputs):
        result = [list(zip(*lists)) for lists in zip(*batched_inputs)]
        return result

    def invoke_function_on_inputs(self, input_args):
        """
        Given a list of inputs, invoke the black-box function on each of them.
        Note that function may fail on some inputs, and we skip those.
        """
        for r in input_args:
            try:
                if not self.caching:
                    yield self.f(*r)
                else:
                    hashable_fn_input = util.get_hashable_elem(r)
                    if hashable_fn_input in self.fn_cache:
                        yield self.fn_cache[hashable_fn_input]
                    else:
                        y = self.timeout_decorator(self.f)(*r)
                        self.fn_cache[hashable_fn_input] = y
                        yield y
            except:
                yield constants.RESERVED_FAILURE
        
    def invoke_function_on_batched_inputs(self, batched_inputs):
        return list(map(self.process_batch, batched_inputs))
    
    def process_batch(self, batch):
        return list(self.invoke_function_on_inputs(batch))
