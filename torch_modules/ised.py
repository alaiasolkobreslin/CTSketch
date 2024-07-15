import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import signal
import functools
import os
import errno

from src import constants
from src import util

class ISED(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.f = kwargs['bbox']
        self.k = kwargs['n_samples']
        self.semiring = kwargs['semiring']
        self.input_mappings = kwargs['input_mappings']
        self.output_mapping = kwargs['output_mapping']
        self.fn_cache = {}
        self.caching = True
        self.timeout_seconds = 1
        self.error_message = os.strerror(errno.ETIME)
        
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
        
    def forward(self, *inputs):
        batch_size = inputs[0].tensor.size(0)
        
        # Prepare the inputs to the black-box function
        to_compute_inputs, sampled_indices = [], []
        for (input_i, input_mapping_i) in zip(inputs, self.input_mappings):
            _, sampled_indices_i, sampled_elements_i = input_mapping_i.sample(
                input_i, sample_count=self.k)
            to_compute = sampled_elements_i
            to_compute_inputs.append(to_compute)
            sampled_indices.append(sampled_indices_i)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)
        
        results = self.invoke_function_on_batched_inputs(to_compute_inputs)
        
        # Aggregate probabilities
        result_probs = torch.ones((batch_size, self.k))
        for (input, idx) in zip(inputs, sampled_indices):
            expanded_input = input.tensor.unsqueeze(0).expand(self.k, -1, -1).transpose(0, 1)
            gathered_probs = torch.gather(expanded_input, 2, idx.unsqueeze(-1)).squeeze(-1) #.t()
            if self.semiring == "add-mult":
                result_probs *= gathered_probs
            elif self.semiring == "min-max":
                result_probs = torch.minimum(
                    result_probs.clone(), gathered_probs)
            else:
                raise Exception("Unknown semiring")
        
        return self.output_mapping.vectorize(results, result_probs)
       
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
