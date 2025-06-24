from math import pi
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

import numpy as np
from .utils import default
import pdb
""" Distributions """

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin

""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)
    
""" Diffusion Methods """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""

    pass


class VDiffusion(Diffusion):
    def __init__(
        self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution()
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)

class StyleVDiffusion(Diffusion):
    def __init__(
        self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution()
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, y: Tensor, x: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device) #(batch)
        sigmas_batch = extend_dim(sigmas, dim=y.ndim) #(batch,1,1,1)
        # Get noise
        noise = torch.randn_like(y)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        y_noisy = alphas * y + betas * noise
        y_noisy = torch.concat((y_noisy, x), dim=1)
        v_target = alphas * noise - betas * y
        # Predict velocity and return loss
        v_pred = self.net(y_noisy, sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)

class StyleMMVDiffusion(Diffusion):
    def __init__(
        self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution()
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, y: Tensor, x: Tensor, img_feat: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device) #(batch)
        sigmas_batch = extend_dim(sigmas, dim=y.ndim) #(batch,1,1,1)
        # Get noise
        noise = torch.randn_like(y)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        y_noisy = alphas * y + betas * noise
        y_noisy = torch.concat((y_noisy, x), dim=1)
        v_target = alphas * noise - betas * y
        # Predict velocity and return loss
        v_pred = self.net(y_noisy, sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)

class BBDMDiffusion(Diffusion):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        m_min, m_max = 0.001, 0.999
        self.schedule = LinearSchedule(start = m_min, end = m_max)
        self.training_num_steps = 1000

    def q_sample(self, x_start, y_end, t, noise=None):

        mean_t = self.schedule(self.training_num_steps, x_start.device)
        variance_t = 2. * (mean_t - mean_t ** 2) * 1.0

        m_t = mean_t[t]
        var_t = variance_t[t]
        sigma_t = torch.sqrt(var_t)

        x_t = (1. - m_t)[...,None,None,None] * x_start + m_t[...,None,None,None] * y_end + sigma_t[...,None,None,None] * noise
        objective = m_t[...,None,None,None] * (y_end - x_start) + sigma_t[...,None,None,None] * noise

        return m_t, x_t, objective

    def forward(self, x: Tensor, target_condition: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Get noise
        t = torch.randint(0, self.training_num_steps, (batch_size,), device=device)
        noise = torch.randn_like(x)
        m_t, x_t, objective = self.q_sample(x, target_condition, t, noise=noise)

        # Predict velocity and return loss
        #v_pred = self.net(torch.cat([x_t,target_condition],dim=1), m_t, **kwargs)
        v_pred = self.net(x_t, m_t, **kwargs)

        return F.mse_loss(v_pred, objective)




class MIDSBDiffusion(Diffusion):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.t_max = 1
        self.t_min = 3.0e-2
        self.beta = 2.0e-2

    def marginal_log_alpha(self, t):
        return - 0.5 * t * self.beta

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_alpha(t))
    
    def marginal_log_sigma(self, t):
        return 0.5 * torch.log(1. - torch.exp(2. * self.marginal_log_alpha(t)))
    
    def marginal_sigma(self, t):
        return torch.exp(self.marginal_log_sigma(t))
    
    def marginal_lambda(self, t):
        return self.marginal_log_alpha(t) - self.marginal_log_sigma(t)
    
    def h(self, s, t):
        return self.marginal_lambda(s) - self.marginal_lambda(t)
    
    def q_sample(self, t, x0, x1):
        m = torch.exp(2.0 * self.h(torch.ones_like(t, device=x0.device)*self.t_max, t))
        mu_xT, mu_x0 = m * self.marginal_alpha(t) / self.marginal_alpha(torch.ones_like(t, device=x0.device) * self.t_max), (1 - m) * self.marginal_alpha(t)
        var = self.marginal_sigma(t) ** 2 * (1 - m)

        mu_x0 = extend_dim(mu_x0, x0.ndim)
        mu_xT = extend_dim(mu_xT, x0.ndim)
        var = extend_dim(var, x0.ndim)

        mean = mu_xT * x1 + mu_x0 * x0
        x_t = mean + var.sqrt() * torch.randn_like(mean)

        return x_t

    def compute_label(self, t, x0, xt):
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)
        alpha_t = extend_dim(alpha_t, x0.ndim)
        sigma_t = extend_dim(sigma_t, x0.ndim)
        label = (xt - x0 * alpha_t) / sigma_t

        return label

    def forward(self, x: Tensor, target_condition: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        
        timestep = torch.rand(batch_size, device=device) * (self.t_max - self.t_min) + self.t_min
        x_t = self.q_sample(t=timestep, x0=x, x1=target_condition)
        label = self.compute_label(t=timestep, x0=x, xt=x_t)

        #score = self.net(torch.cat([x_t,target_condition],dim=1), timestep, **kwargs)
        score = self.net(x_t, timestep, **kwargs)

        return F.mse_loss(score, label)


class DOSEDiffusion(Diffusion):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.training_num_steps = 50
        noise_schedule = np.linspace(1e-4, 0.035, self.training_num_steps)
        noise_level = np.cumprod(1 - noise_schedule)
        self.noise_level = torch.Tensor(noise_level.astype(np.float32))
        self.dropout = 0.5

    def forward(self, x: Tensor, target_condition: Tensor, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device
        self.noise_level = self.noise_level.to(device)
        t = torch.randint(0, self.training_num_steps, [batch_size], device=device)

        noise_scale = self.noise_level[t]
        noise_scale = extend_dim(noise_scale,x.ndim)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(x)

        x_t = noise_scale_sqrt * x + (1.0 - noise_scale)**0.5 * noise
        predicted = self.net(torch.cat([x_t,target_condition],dim=1), t, **kwargs)
        return F.mse_loss(x, predicted)


class ARVDiffusion(Diffusion):
    def __init__(self, net: nn.Module, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.net = net
        self.length = length
        self.num_splits = num_splits
        self.split_length = length // num_splits

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Returns diffusion loss of v-objective with different noises per split"""
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        assert t == self.length, "input length must match length"
        # Sample amount of noise to add for each split
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        sigmas = repeat(sigmas, "b 1 n -> b 1 (n l)", l=self.split_length)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Sigmas will be provided as additional channel
        channels = torch.cat([x_noisy, sigmas], dim=1)
        # Predict velocity and return loss
        v_pred = self.net(channels, **kwargs)
        return F.mse_loss(v_pred, v_target)


""" Samplers """


class Sampler(nn.Module):
    pass


class VSampler(Sampler):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy
    
class StyleVSampler(Sampler):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy:Tensor, x: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=show_progress)

        for i in progress_bar:
            x_mix = torch.cat((x_noisy, x), dim=1)
            v_pred = self.net(x_mix, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy

class StyleMMVSampler(Sampler):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy:Tensor, x: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=show_progress)

        for i in progress_bar:
            x_mix = torch.cat((x_noisy, x), dim=1)
            v_pred = self.net(x_mix, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy

class BBDMSampler(Sampler):

    diffusion_types = [BBDMDiffusion]

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        m_min, m_max = 0.001, 0.999
        self.schedule = LinearSchedule(start = m_min, end = m_max)
        self.training_num_steps = 1000
        self.sample_steps = 200

        midsteps = torch.arange(self.training_num_steps - 1, 1, step=-((self.training_num_steps - 1) / (self.sample_steps - 2))).long()
        self.indices = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor, target_condition: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        
        mean_t = self.schedule(self.training_num_steps, x_noisy.device)
        variance_t = 2. * (mean_t - mean_t ** 2) * 1.0
        progress_bar = tqdm(range(self.sample_steps), disable=show_progress)

        for i in progress_bar:
            if self.indices[i] == 0:
                t = torch.full((x_noisy.shape[0],), self.indices[i], device=x_noisy.device, dtype=torch.long)
                #v_pred = self.net(torch.cat([x_noisy,target_condition],dim=1), mean_t[t], **kwargs)
                v_pred = self.net(x_noisy, mean_t[t], **kwargs)
                x0_recon = x_noisy - v_pred
                x_noisy = x0_recon
                
            else:
                t = torch.full((x_noisy.shape[0],), self.indices[i], device=x_noisy.device, dtype=torch.long)
                n_t = torch.full((x_noisy.shape[0],), self.indices[i+1], device=x_noisy.device, dtype=torch.long)
                #v_pred = self.net(torch.cat([x_noisy, target_condition],dim=1), mean_t[t], **kwargs)
                v_pred = self.net(x_noisy, mean_t[t], **kwargs)
                x0_recon = x_noisy - v_pred

                m_t = mean_t[t]
                m_nt = mean_t[n_t]
                m_t = extend_dim(m_t, dim=x_noisy.ndim)
                m_nt = extend_dim(m_nt, dim=x_noisy.ndim)
                var_t = variance_t[t]
                var_nt = variance_t[n_t]
                var_t = extend_dim(var_t, dim=x_noisy.ndim)
                var_nt = extend_dim(var_nt, dim=x_noisy.ndim)

                sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
                sigma2_t = extend_dim(sigma2_t, dim=x_noisy.ndim)
                sigma_t = torch.sqrt(sigma2_t) * 1.0


                noise = torch.randn_like(x_noisy)
                x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * target_condition + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                                (x_noisy - (1. - m_t) * x0_recon - m_t * target_condition)                
                x_noisy = x_tminus_mean + sigma_t * noise

        return x_noisy
    
class MIDSBSampler(Sampler):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.sample_steps = 200
        self.t_max = 1
        self.t_min = 3.0e-2
        self.beta = 2.0e-2

    def marginal_log_alpha(self, t):
        return - 0.5 * t * self.beta

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_alpha(t))
    
    def marginal_log_sigma(self, t):
        return 0.5 * torch.log(1. - torch.exp(2. * self.marginal_log_alpha(t)))
    
    def marginal_sigma(self, t):
        return torch.exp(self.marginal_log_sigma(t))
    
    def marginal_lambda(self, t):
        return self.marginal_log_alpha(t) - self.marginal_log_sigma(t)
    
    def h(self, s, t):
        return self.marginal_lambda(s) - self.marginal_lambda(t)
    
    @torch.no_grad()
    def compute_pred_x0(self, t, xt, net_out, clip_denoise=False):
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)        
        alpha_t = extend_dim(alpha_t, xt.ndim)
        sigma_t = extend_dim(sigma_t, xt.ndim)
        pred_x0 = (xt - sigma_t * net_out) / alpha_t

        return pred_x0

    @torch.no_grad()
    def pred_x0_fn(self, xt, target_condition, timestep, **kwargs):
        #global NFE
        timestep = torch.full((xt.shape[0],), timestep, device=xt.device, dtype=torch.float32)
        #out = self.net(torch.cat([xt,target_condition],dim=1), timestep, **kwargs)
        out = self.net(xt, timestep, **kwargs)
        #NFE = NFE + 1
        return self.compute_pred_x0(t=timestep, xt=xt, net_out=out)
    
    def p_posterior(self, t, s, x, x0, ot_ode=False):
        m = torch.exp(2.0 * self.h(s, t))
        mu_xt, mu_x0 = m * self.marginal_alpha(t) / self.marginal_alpha(s), (1 - m) * self.marginal_alpha(t)

        mu_xt = extend_dim(mu_xt, x.ndim)
        mu_x0 = extend_dim(mu_x0, x.ndim)

        xt_prev = mu_x0 * x0 + mu_xt * x

        if not ot_ode and t > self.t_min:
            var = self.marginal_sigma(t) ** 2 * (1 - m)
            var = extend_dim(var, x.ndim)
            xt_prev += var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev
    
    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor, target_condition: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        
        # time_uniform
        x = x_noisy
        timesteps = torch.linspace(self.t_max, self.t_min, self.sample_steps + 1, device=x_noisy.device)          
        for i in range(0, self.sample_steps):
            t, t_prev = timesteps[i], timesteps[i+1]
            pred_x0 = self.pred_x0_fn(x, target_condition, t, **kwargs)
            x = self.p_posterior(t_prev, t, x, pred_x0)

        return x

        
class DOSESampler(Sampler):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.training_num_steps = 50
        noise_schedule = np.linspace(1e-4, 0.035, self.training_num_steps)
        self.training_noise_schedule = noise_schedule
        self.inference_noise_schedule = self.training_noise_schedule

        talpha = 1 - self.training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        alpha = 1 - self.inference_noise_schedule
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(self.inference_noise_schedule)):
            for t in range(len(self.training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        self.T = np.array(T, dtype=np.float32)

        self.noise_level = torch.Tensor(talpha_cum.astype(np.float32))

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor, target_condition: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        device = x_noisy.device
        time_step = 35
        _step = torch.full([1],time_step)
        noise_scale = self.noise_level[_step].to(device)
        noise_scale = extend_dim(noise_scale, x_noisy.ndim)
        noise_scale_sqrt = noise_scale ** 0.5

        noise = torch.randn_like(x_noisy).to(device)
        audio = noise_scale_sqrt * x_noisy + (1.0 - noise_scale)**0.5 * noise
        audio = self.net(torch.cat([audio,target_condition],dim=1), torch.tensor([self.T[time_step]],device=device), **kwargs)
        audio = torch.clamp(audio, -1.0, 1.0)

        time_step = 15
        audio = 0.5 * (audio + target_condition)
        _step = torch.full([1], time_step)
        noise_scale = self.noise_level[_step].to(device)
        noise_scale = extend_dim(noise_scale, x_noisy.ndim)
        noise_scale_sqrt = noise_scale ** 0.5

        noise = torch.randn_like(x_noisy).to(device)
        audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
        audio = self.net(torch.cat([audio,target_condition],dim=1), torch.tensor([self.T[time_step]],device=device), **kwargs)
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio


class ARVSampler(Sampler):
    def __init__(self, net: nn.Module, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
        self, current: Tensor, sigmas: Tensor, show_progress: bool = False, **kwargs
    ) -> Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            channels = torch.cat([current, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return current

    def sample_start(self, num_items: int, num_steps: int, **kwargs) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Same sigma schedule over all chunks
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        # Return start if only num_splits chunks
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))

        # Loop over ladder shifts
        num_shifts = num_chunks  # - self.num_splits
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)

        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs
            )
            # Update chunks
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            shape = (b, self.in_channels, self.split_length)
            chunks += [torch.randn(shape, device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)


"""  Inpainters """


class Inpainter(nn.Module):
    pass


class VInpainter(Inpainter):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self,
        source: Tensor,
        mask: Tensor,
        num_steps: int,
        num_resamples: int,
        show_progress: bool = False,
        x_noisy: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        x_noisy = default(x_noisy, lambda: torch.randn_like(source))
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            for r in range(num_resamples):
                v_pred = self.net(x_noisy, sigmas[i], **kwargs)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                # Renoise to current noise level if resampling
                j = r == num_resamples - 1
                x_noisy = alphas[i + j] * x_pred + betas[i + j] * noise_pred
                s_noisy = alphas[i + j] * source + betas[i + j] * torch.randn_like(
                    source
                )
                x_noisy = s_noisy * mask + x_noisy * ~mask

            progress_bar.set_description(f"Inpainting (noise={sigmas[i+1,0]:.2f})")

        return x_noisy
