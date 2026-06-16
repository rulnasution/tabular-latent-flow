import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from typing import *
# from tab_transformer_pytorch import TabTransformer

#@title ⏳ Summary: please run this cell which contains the ```VPDiffusionFlowMatching``` class

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

class FlowMatching:

    def __init__(self, target='velocity') -> None:
        super().__init__()
        
        '''
        matching: the calculation of target velocity/score (see https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
            - "velocity": follows original FM velocity calculation (equation 21, example 11)
            - "velocity_x0": follows alternative form v_t = alpha_dot(t) x1 + beta_dot(t) x0 (see Example 19)
            - "score": score matching (See example 22)
            - "score_x0": score matching with alternate form (See example 22)
            - "ddpm": noise matching as in DDPM
        '''
        assert target in ['velocity', 'velocity_x0', 'score', 'score_x0', 'ddpm'], f'target "{target}" does not exist'
        self.eps = 1e-5
        self.target = target

        
    def loss(self, v_t: nn.Module,
           x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        t = torch.rand((x_1.shape[0], 1), device=x_1.device) % (1 - self.eps)
        x_0 = torch.randn_like(x_1)
        
        (alpha, beta), (dalpha, dbeta) = t_dir(v_t.net_t, t)
        
        alpha = alpha.expand(x_1.shape)
        beta = beta.expand(x_1.shape)
        dalpha = dalpha.expand(x_1.shape)
        dbeta = dbeta.expand(x_1.shape)

        x_t = alpha * x_1 + beta * x_0
        approximate = v_t(t[:,0], x_t)

        ### Calculate target velocity/score
        if self.target == 'velocity':
            p1 = dalpha - (dbeta/beta) * alpha
            p2 = dbeta/beta
            field_target = p1 * x_1 + p2 * x_t
        elif self.target == 'velocity_x0':
            field_target = dalpha * x_1 + dbeta * x_0
        elif self.target == 'score':
            '''
            https://jmtomczak.github.io/blog/16/16_score_matching.html and 
            https://openreview.net/pdf?id=PxTIG12RRHS,
            example colab from google research: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=zOsoqPdXHuL5
            set lambda_t as beta**2 -> therefore, inside will be sqrt(lambda(t))
            in Lipman's paper:
            Taking λ(t) = σ_t**2(x1) corresponds to the original Score Matching (SM) loss from Song & Ermon
            (2019), while considering λ(t) = β(1−t) corresponds to the Score Flow (SF)
            loss motivated by an NLL upper bound (Song et al., 2021); 
            '''
            approximate /= beta
            field_target = -(x_t - alpha*x_1) / beta ** 2
        elif self.target == 'score_x0':
            approximate /= beta
            field_target = - x_0
            # field_target = - x_0 / beta
        elif self.target == 'ddpm':
            field_target = x_0

        if 'velocity' in self.target or self.target == 'ddpm':
            weight = torch.ones_like(beta) 
        else:
            weight = beta ** 2
            
        return torch.mean(weight * (approximate - field_target)**2)
        # return torch.mean((approximate * beta + x_0)**2)

class SiLU(nn.Module):
        def forward(self, x):
                return x * torch.sigmoid(x)

class Net(nn.Module): ## followed from tabsyn paper
    def __init__(self, in_dim: int, n_frequencies:int) -> None:
        super().__init__()

        dim_t = 2 * n_frequencies
        ins = [dim_t, dim_t*2, dim_t*2]
        outs = [dim_t*2, dim_t*2, dim_t]
        
        self.n_frequencies = n_frequencies

        self.proj = nn.Linear(in_dim, dim_t)

        self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(in_d, out_d), nn.SiLU()) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(dim_t, in_dim))

        self.time_embed = nn.Sequential(
                nn.Linear(2 * n_frequencies, 2 * n_frequencies),
                nn.SiLU(),
                nn.Linear(2 * n_frequencies, 2 * n_frequencies)
        )

    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t[..., None]
        return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.time_encoder(t)
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        for l in self.layers:
            x = l(x)
        return self.top(x)


### the conditional vector fields baseline
class CondVF(nn.Module):
    def __init__(self, net: nn.Module, net_t: nn.Module, n_steps: int = 100, score = False, ddpm=False) -> None:
        super().__init__()
        self.net = net
        self.net_t = net_t
        self.useodeint = False
        self.n_steps = n_steps
        self.nfe = 0
        self.score = score
        self.ddpm = ddpm
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # print(t.size(), x.size())
        if self.useodeint:
            return self.forward_for_ode(t, x)
        else:
            return self.net(t, x)
    
    def forward_for_ode(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        t = t.expand(x.size(0))
        
        approx_field = self.net(t, x)
        if not self.score: 
            return approx_field  # return velocity langsung
        
        # Conversion dari score ke velocity
        t = t[:, None].expand((x.size(0), 1))
        
        (alpha, beta), (dalpha, dbeta) = t_dir(self.net_t, t)
        
        alpha = alpha.expand(x.shape)
        beta = beta.expand(x.shape)

#         eps = 1e-4
#         alpha = torch.clamp(alpha, min=eps, max=1-eps).expand(x.shape)
#         beta = torch.clamp(beta, min=eps, max=1-eps).expand(x.shape)
    
        dalpha = dalpha.expand(x.shape)
        dbeta = dbeta.expand(x.shape)
        
        # Convert noise → score (kalau ddpm=True)
        if self.ddpm: approx_field = -approx_field / beta  # approx_field sekarang score
        else: approx_field = approx_field / beta ## for SM it is still raw

        # Restructure untuk stability: hindari subtraksi besar
        # p1 = β² (dα/α) - β dβ = β[β(dα/α) - dβ]
        alpha_ratio = dalpha / alpha
        p1 = beta * (beta * alpha_ratio - dbeta)
        p2 = alpha_ratio

        # print(torch.cat([p1, p2], dim=1)[:5,:])
        
#         p1 = torch.clamp(p1, min=-100, max=100)
#         p2 = torch.clamp(p2, min=-100, max=100)
    
        return p1 * approx_field + p2 * x
  
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
                 n_steps: int = 100, score = True, ddpm=True) -> None:
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
        self.nfe = 0
        self.score = score
        self.ddpm = ddpm
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        ### Drift term for SDE: f(x,t) + 0.5 * sigma(t)^2 \nabla_x log p(x_t)
        # print(t)
        t = t.expand(x.size(0))

        ## collect relevant object
        if self.score: score = self.net(t, x) 
        else: vtx = self.net(t, x) 
        
        t = t[:, None].expand((x.size(0),1))
        
        (alpha, beta), (dalpha, dbeta) = t_dir(self.net_t, t)
        alpha = alpha.expand(x.shape)
        beta = beta.expand(x.shape)
        dalpha = dalpha.expand(x.shape)
        dbeta = dbeta.expand(x.shape)
        
        # print(alpha.max().item(), beta.max().item(), dalpha.max().item(), dbeta.max().item())

        if self.score:
#             eps = 1e-4
#             alpha = torch.clamp(alpha, min=eps, max=1-eps).expand(x.shape)
#             beta = torch.clamp(beta, min=eps, max=1-eps).expand(x.shape)
            if self.ddpm: score = -score/beta
            else: score = score / beta
            
            alpha_ratio = dalpha / alpha
            p1 = beta * (beta * alpha_ratio - dbeta)
            p2 = alpha_ratio
            # print(p1.max().item(), p2.max().item())

            
            # Optional: clip intermediate results
#             p1 = torch.clamp(p1, min=-100, max=100)
#             p2 = torch.clamp(p2, min=-100, max=100)
            vtx = p1 * score + p2 * x
        else: ## score conversion from velocity (see eq 55 of https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 
            p1 = (alpha * vtx) - (dalpha * x)
            p2 = (beta**2 * dalpha) - (alpha * dbeta * beta)
            score = p1/p2
        
        '''
        The additional part for SDE, sigma and score are according to
        Example 14 (pp. 20), Summary 17 (pp. 22)
        https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
        (Section 3.1 pp. 14) Remember that in the lecture, z~p_data.
        So, z is the data or x_1.
        See also eq. 56 on the same lecture.
        '''

        _, sigma_t = self.net_sigma_t(t)
        gt = self.sigma_max * sigma_t

        ### for sigma_max = 0 (ODE), no need to clip
        return vtx + 0.5 * gt**2 * score

    def g(self, t: torch.Tensor, x: torch.Tensor):
        ### Diffusion term for SDE: \sigma(t)dWt
        _, sigma_t = self.net_sigma_t(t)
        gt = self.sigma_max * sigma_t
        return gt.expand(x.size())


#### network for time function ####

class OT_t(nn.Module):
    '''
    just regular t to ensure it works in gaussian setting
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()

    def atx(self, t: torch.Tensor) -> torch.Tensor:
        # return torch.ones_like(t.view(-1,1))
        # return 1 / (1. - (0.999 * t))
        return (1 / (1. - (0.999 * t.view(-1,1))))

        # return t.view(-1,1)
        
    def forward(self, t : Tensor) -> Tensor:
        return t.view(-1,1), 1. - (0.999 * t.view(-1,1))


class SoftOT_t(nn.Module):
    """
    Softened OT Interpolation Path suitable for Score Matching.
    
    - alpha(t) = eps + (1-eps) * t
    - beta(t) = max(1 - t, beta_min)

    Ensures:
        - alpha never becomes 0  -> prevents (dot(alpha)/alpha) blowup
        - beta never becomes 0   -> prevents score = raw/beta overflow
    """

    def __init__(self, eps=0.01, beta_min=0.05):
        super().__init__()
        self.eps = eps
        self.beta_min = beta_min

    # velocity component: dot(alpha)/alpha
    def atx(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes: dot(alpha)/alpha = (1-eps) / (eps + (1-eps)*t)
        This was the main cause of singularity in pure OT.
        """
        t = t.view(-1,1)
        alpha = self.eps + (1 - self.eps) * t
        dot_alpha = (1 - self.eps)
        return dot_alpha / alpha

    def forward(self, t: torch.Tensor):
        """
        Returns (alpha(t), beta(t))
        """

        t = t.view(-1,1)

        # softened alpha
        alpha = self.eps + (1 - self.eps) * t

        # # softened beta
        # beta = 1 - t
        # beta = torch.clamp(beta, min=self.beta_min)

        return alpha, 1. - (0.999 * t)

class VPDiffusion_t(nn.Module):
    '''
    Variance preserving diffusion field
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5

    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t*(self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.T(t))
    
    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - self.alpha(1. - t) ** 2)

    def atx(self, t: torch.Tensor) -> torch.Tensor:
        num = - torch.exp(-0.5 * self.T(1.-t)) ## alpha (t)
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(1. - t), torch.sqrt(1. - self.alpha(1. - t) ** 2)

class VEDiffusion_t(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 1.
        self.eps = 1e-5


    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
    
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
    
        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max/self.sigma_min))
    
    def atx(self, t: torch.Tensor) -> torch.Tensor:
        return (self.dsigma_dt(1. - t) / self.sigma_t(1. - t))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t), self.sigma_t(1. - t)
    
class VEDiffusion2_t(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 1.
        self.eps = 1e-5

    def atx(self, t: torch.Tensor) -> torch.Tensor:
        return -1. / (1. - t)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t), 1. - t
    
class LogitNormal_t(nn.Module):
    '''
    Logit normal distribution-based time function
    See FLUX.1 paper for details: https://arxiv.org/pdf/2506.15742
    I put 0.999 for stability purposes.
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()
        self.alpha = 3.0
        self.sigma = 1.0

    def atx(self, t):
        t = t.clamp(min=1e-7)
        
        denum_add = (1 / (0.999 * t.view(-1,1)) - 1)**self.sigma
        t_prime = self.alpha / (self.alpha + denum_add) 

        num = self.sigma * self.alpha * (1/(0.999 * t.view(-1,1)) - 1)**(self.sigma - 1)
        den = 0.999 * t**2 * (self.alpha + (1/(0.999 * t.view(-1,1)) - 1)**self.sigma)**2
        # num = self.sigma * self.alpha * (1/(0.999 * t.view(-1,1)) - 1)**(self.sigma - 1)
        t_prime_dot = num/den

        return t_prime_dot / (1 - t_prime)
        
    def forward(self, t : Tensor) -> Tensor:
        t = t.clamp(min=1e-7)
        denum_add = (1 / (0.999 * t.view(-1,1)) - 1)**self.sigma
        # denum_add = (1 / t.view(-1,1) - 1)**self.sigma
        t_prime = self.alpha / (self.alpha + denum_add)  
        return t_prime, 1. - t_prime

class Cosine_t(nn.Module):
    '''
    cosine time function
    See stochastic interpolant paper for details: https://arxiv.org/pdf/2209.15571
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()

    def atx(self, t) -> Tensor:
        t = t.clamp(max=1 - 1e-5)
        alpha, beta = torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)
        dalpha = torch.cos(0.5 * math.pi * t) * 0.5 * math.pi
        dbeta = -torch.sin(0.5 * math.pi * t) * 0.5 * math.pi
        # (alpha, beta), (dalpha, dbeta) = t_dir(self, t)
        return dalpha - (dbeta/beta) * alpha
        
    def forward(self, t : Tensor) -> Tensor:
        
        return torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)