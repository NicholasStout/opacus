from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, DPOptimizer
from torch.distributions.laplace import Laplace
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch import stack, zeros, einsum
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer

class PLRVDPOptimizer(DPOptimizer):
    """
    Implementation of PLRV first noise mechanism.
    """
    
    
    def add_noise(self):
        self.k = 2
        self.theta = 2
        self.mu = 1
        self.sigma = 1
        self.a = 1
        self.b = 1.1
        self.gamma = Gamma(
          concentration = self.k, rate = self.theta
          )
        self.normal= Normal(
          loc = self.mu, scale = self.sigma
          )
        self.uniform = Uniform(
          low = self.a, high = self.b
          )
          
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            laplace = self.get_laplace()
            noise = laplace.sample(p.summed_grad.shape).to(p.summed_grad.device)
            p.grad = p.summed_grad + noise

            _mark_as_processed(p.summed_grad)
            
    def get_linear_combination(self):
        #gauss_sample = self.normal.sample()
        #if gauss_sample < self.l:
        #  gauss_sample = self.l
        #elif gauss_sample < self.u:
        #  gauss_sample = self.u
          
        return (self.gamma.sample())#+self.normal.sample()+self.uniform.sample())
        
    def get_laplace(self):
        return Laplace(loc=0, scale=self.get_linear_combination() * self.max_grad_norm)
        
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(1, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
