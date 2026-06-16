import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import math
# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import TensorDataset
from typing import *

def jvp(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[torch.Tensor], torch.Tensor ], t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return jvp(f, t, torch.ones_like(t))

def get_t_dir(model: nn.Module, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def f(t_in):
        def f_(t_in):
            return model(t_in)
        return f_

    return t_dir(f(t), t)

class ContVarFlowMatching:

    def __init__(self, var_weight='efvfm') -> None:
        super().__init__()
        
        '''
        var_weight: time weight used for gaussian loss, can be used as variance.
            - "theory": follows theorem 3 of Eijkelboom et al. (using 2atx^2 as variance)
            - "efvfm": follows implementation of efvfm (guzman cordero et al.).
            - "atx": follow the implementation of tabvvfm (using 2atx as variance)
            - "learnable": uses predicted posterior variance, dimension wise (ice simulation paper) 
        '''
        # assert var_weight in ['theory', 'efvfm', 'atx','learnable']
        self.eps = 1e-5
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.var_weight = var_weight

    
    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, alpha: torch.Tensor, beta: torch) -> torch.Tensor:
        """ Conditional Flow
        """
        return alpha * x_1 + beta * x ## for continuous only until here
        
    def loss(self, v_t: nn.Module,
           x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        # t = (torch.rand(1, device=x_1.device) + torch.arange(x_1.shape[0], device=x_1.device) / x_1.shape[0]) % (1 - self.eps)
        # t = t.view(-1, 1)
        t = torch.rand((x_1.shape[0], 1), device=x_1.device) % (1 - self.eps)
        # t = t[:, None].expand(x_1.shape)
        # print(t.size())
        x_0 = torch.randn_like(x_1)
        # (alpha, beta), (dalpha, dbeta) = t_dir(v_t.net_t, 0.999 * t)
        alpha, beta = v_t.net_t(t)
        atx = v_t.net_t.atx(t).view(-1,1)

        alpha = alpha.expand(x_1.shape)
        beta = beta.expand(x_1.shape)
        atx = atx.expand(x_1.shape)
        
        x_t= self.psi_t(x_0, x_1, alpha, beta)
        
        theta_all = v_t(t[:,0], x_t)
        
        identity = torch.eye(x_1.shape[1], device=x_1.device, dtype=torch.float).unsqueeze(0).expand(x_1.shape[0], -1, -1)
        if self.var_weight == 'atx':
            normal_dist = Normal(theta_all, 1/(torch.sqrt(2*atx)))
            return torch.mean(-normal_dist.log_prob(x_1))
        elif self.var_weight == 'atx_sum':
            normal_dist = Normal(theta_all, 1/(torch.sqrt(2*atx)))
            return -normal_dist.log_prob(x_1).sum(dim=1).mean()
        elif self.var_weight == 'mse':
            return torch.mean((x_1-theta_all)**2)
        elif self.var_weight == 'mse_sum':
            normal_dist = Normal(theta_all, 1/(torch.sqrt(2*atx)))
            return torch.mean(((x_1-theta_all)**2).sum(dim=1))
        elif self.var_weight == 'efvfm':
            scale = 1 - (1 - 0.01) * t.unsqueeze(1) ** 2
            sigma = scale * identity
            dist = MultivariateNormal(theta_all, sigma)
            return -dist.log_prob(x_1).mean()
        elif self.var_weight == 'efvfm_atx':
            scale =  (1 + math.sqrt(1 - 0.001) * t) / atx[:,0].view(-1,1)
            sigma = scale.unsqueeze(1) * identity
            dist = MultivariateNormal(theta_all, sigma)
            return -dist.log_prob(x_1).mean()
        elif self.var_weight == 'theory':
            # '''Σt(x) = 1/2 (A^T_t(x)At(x))^{-1}'''
            A_t = torch.diag_embed(atx)  # Shape: (batch_size, d_cont, d_cont)
            AtT_At = torch.matmul(A_t.transpose(1, 2), A_t)       # (batch_size, d_cont, d_cont)
            sigma = 0.5 * torch.linalg.inv(AtT_At)               # (batch_size, d_cont, d_cont)
            dist = MultivariateNormal(theta_all, sigma)
            return -dist.log_prob(x_1).mean()
        elif self.var_weight == 'theory_mean':
            normal_dist = Normal(theta_all, 1/(math.sqrt(2.)*atx))
            return -normal_dist.log_prob(x_1).mean()
        elif self.var_weight == 'learnable':
            theta_all = torch.chunk(theta_all, 2, dim=1) ## mean, variance
            normal_dist = Normal(theta_all[0], F.softplus(theta_all[1]))
            return -normal_dist.log_prob(x_1).sum(dim=1).mean()

        # dist = MultivariateNormal(theta_all, sigma)
        # loss_q = -dist.log_prob(x_1).mean()
        
        # return loss_q


### the conditional vector fields baseline
class CondVF(nn.Module):
    def __init__(self, net: nn.Module, net_t: nn.Module, n_steps: int = 100) -> None:
        super().__init__()
        self.net = net
        self.net_t = net_t
        self.atol=1e-06
        self.rtol=1e-05
        self.useodeint = False
        self.n_steps = n_steps
        self.learnable = False
        self.nfe = 0
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # print(t.size(), x.size())
        if self.useodeint:
            return self.forward_for_ode(t, x)
        else:
            return self.net(t, x)
    
    def forward_for_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        t = t.expand(x.size(0))
        x1 = self.net(t, x) ## x1 = theta_t(x_t) in the equation
        if self.learnable: x1 = torch.chunk(x1, 2, dim=1)[0] ## just get the mean
        t = t[:, None].expand((x.size(0),1))
    
        (alpha, beta), (dalpha, dbeta) = t_dir(self.net_t, t)
        alpha = alpha.expand(x1.shape)
        beta = beta.expand(x1.shape)
        dalpha = dalpha.expand(x1.shape)
        dbeta = dbeta.expand(x1.shape)

        p1 = dalpha - (dbeta/beta) * alpha
        p2 = dbeta/beta
        return p1 * x1 + p2 * x
  
    def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)
    
    # uses from torchdiffeq import odeint_adjoint as odeint
    def decode_t0_t1(self, x_0, t0, t1, method='euler'):
        self.useodeint = True
        
        # Default: step_size untuk fixed-step, rtol/atol untuk adaptive
        if method in ['euler', 'rk4', 'midpoint']:
            options = {'step_size': 1 / self.n_steps}
        # Untuk solver adaptive (dopri5, adams, dll) bisa set rtol/atol kalau perlu:
        elif method in ['dopri5', 'adaptive_heun', 'explicit_adams', 'implicit_adams']:
            options = {}
            # 'rtol': 1e-5, 'atol': 1e-6
    
        x_1 = odeint(
            self, 
            y0=x_0, 
            t=torch.tensor([t0, t1], device=x_0.device),
            method=method, 
            options=options,
            rtol=1e-5, atol=1e-6
        )[-1]
        
        self.useodeint = False
        return x_1
    
    def encode(self, x_1: torch.Tensor) -> torch.Tensor:
        self.useodeint = True
        x_0 = odeint(self, y0=x_1, t=torch.tensor([1., 0.],device=x_1.device), 
                      method='euler', options={'step_size': (1/self.n_steps)})[-1]
        self.useodeint = False
        return x_0
    
    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        self.useodeint = True
        x_1 = odeint(self, y0=x_0, t=torch.tensor([0., 1.],device=x_0.device), 
                      method='euler', options={'step_size': (1/self.n_steps)})[-1]
        self.useodeint = False
        return x_1
    
    def decode_manual(self, x_0: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
        x_1 = torch.zeros_like(x_0)
        tt = torch.tensor(0., device=x_0.device)
        h = 1/n_steps
        for _ in range(self.n_steps):
            x_1 += h * self.forward_for_ode(tt, x_1)
            tt += h
        return x_1

### the conditional vector fields using SDE
### documentation can be seen in https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md
class StochasticCondVF(nn.Module):
    def __init__(self, net: nn.Module, net_t: nn.Module, net_sigma_t: nn.Module, sigma_max: float = 1.,
                 n_steps: int = 100) -> None:
        super().__init__()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        
        self.net = net
        self.net_t = net_t
        self.net_sigma_t = net_sigma_t
        self.atol=1e-06
        self.rtol=1e-05
        self.useodeint = False
        self.n_steps = n_steps
        self.sigma_max = sigma_max ### equal to multiplier of the sigma_t
        self.learnable = False
        self.nfe = 0
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        ### Drift term for SDE: f(x,t) + 0.5 * sigma(t)^2 \nabla_x log p(x_t)
        # print(t)
        t = t.expand(x.size(0))
        # print(t.size())
        x1 = self.net(t, x) ## x1 = theta_t(x_t) in the equation
        if self.learnable: x1 = torch.chunk(x1, 2, dim=1)[0] ## just get the mean
        t = t[:, None].expand((x.size(0),1))
    
        (alpha, beta), (dalpha, dbeta) = t_dir(self.net_t, t)
        alpha = alpha.expand(x1.shape)
        beta = beta.expand(x1.shape)
        dalpha = dalpha.expand(x1.shape)
        dbeta = dbeta.expand(x1.shape)
        
        # '''since torchsde cannot overcome the Inf problem when we divide by beta, 
        #    we need to clip beta manually'''
        # print(beta)
        # eps = torch.finfo(beta.dtype).tiny  # ~1e-45 for float32
        # beta = torch.where(beta > eps, beta, eps)
        # dbeta = torch.where(dbeta.abs() > eps, dbeta, torch.zeros_like(beta))
        # if torch.sum(beta == 0) > 0: 
        #     beta = torch.clip(beta, min=1e-3) 

        p1 = dalpha - (dbeta/beta) * alpha
        p2 = dbeta/beta
        vtx = p1 * x1 + p2 * x
        # print(torch.cat([alpha[:1, :2], dalpha[:1, :2], beta[:1, :2], dbeta[:1, :2]], dim=1))
        # print(torch.cat([p1[:1, :2], p2[:1, :2]], dim=1))
        
        '''
        The additional part for SDE, sigma and score are according to
        Example 14 (pp. 20), Summary 17 (pp. 22)
        https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
        (Section 3.1 pp. 14) Remember that in the lecture, z~p_data.
        So, z is the data or x_1.
        '''
        _, sigma_t = self.net_sigma_t(t)
        # score = - (x1 - alpha * x) / beta**2 ## last time we used this before Sept 5 2025
        score = - (x - alpha * x1) / beta**2 ## We will try this instead
        # gt = torch.clip(self.sigma_max * sigma_t, min=0.05) if self.sigma_max > 0 else self.sigma_max * sigma_t
        gt = self.sigma_max * sigma_t
        # a = vtx + 0.5 * gt**2 * score
        # print(torch.cat([vtx[:1, :2], score[:1, :2], gt[:1, :2], a[:1, :2]], dim=1))
        # print((vtx + 0.5 * sigma_t**2 * score).size())

        ### for sigma_max = 0 (ODE), no need to clip
        return vtx + 0.5 * gt**2 * score

    def g(self, t: torch.Tensor, x: torch.Tensor):
        ### Diffusion term for SDE: \sigma(t)dWt
        _, sigma_t = self.net_sigma_t(t)
        # print(sigma_t)
        ### for sigma_max = 0 (ODE), no need to clip
        # gt = torch.clip(self.sigma_max * sigma_t, min=0.05) if self.sigma_max > 0 else self.sigma_max * sigma_t
        gt = self.sigma_max * sigma_t
        return gt.expand(x.size())
        # return sigma_t.expand(x.size())
        # return torch.zeros_like(x)
        