from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, DPOptimizer
from torch.distributions.laplace import Laplace
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch import stack, zeros, einsum
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer
from scipy.stats import truncnorm, expon
from typing import Callable, Optional

class PLRVDPOptimizer(DPOptimizer):
    """
    Implementation of PLRV first noise mechanism.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        
    def make_noise(self, args):
        self.gamma = None
        self.uniform = None
        self.normal = None
        self.args = args
        if 'gamma' in args.keys():
            if args['gamma']:
                self.k = self.args['k']
                self.theta = self.args['theta']
                self.gamma = Gamma(
                    concentration = self.k, rate = self.theta
                )
        if 'uniform' in args.keys():
            if args['uniform']:
                self.a = self.args['a']
                self.b = self.args['b']
                self.uniform = Uniform(
                    low = self.a, high = self.b
                )
        if 'truncnorm' in args.keys():
            if args['truncnorm']:
                self.mu = self.args['mu']
                self.sigma = self.args['sigma']
                self.l = self.args['l']
                self.u = self.args['u']
                a_transformed = (self.l - self.mu) / self.sigma 
                b_transformed = (self.u - self.mu) / self.sigma
                self.normal= truncnorm(
                    a_transformed, b_transformed, loc=self.mu, scale=self.sigma
                )
        
        self.clip = self.args['max_grad_norm']
        #self.lam = self.args['lam']
        
        #self.expon = expon(loc=0, scale = 1/self.lam)
        #self.laplace = self.get_laplace()
    
    
    def add_noise(self):
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            laplace = self.get_laplace()
            noise = laplace.sample(p.summed_grad.shape).to(p.summed_grad.device)
            p.grad = p.summed_grad + noise

            _mark_as_processed(p.summed_grad)
            
    def get_linear_combination(self):
        den = 0
        if self.gamma is not None:
            den += self.gamma.sample()
        if self.uniform is not None:
            den += self.uniform.sample()
        if self.normal is not None:
            den += self.normal.rvs(size=1)[0]  
        #exp = self.expon.rvs(size=1)[0]
        
        return 1/den

        
    def get_laplace(self):
        return Laplace(loc=0, scale=self.get_linear_combination())
        
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
            #print(self.max_grad_norm)
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
