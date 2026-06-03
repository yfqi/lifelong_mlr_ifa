import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as tD
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-1], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp     

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x, train=True):
        
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)   # 把x映射为一个混合高斯分布
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)
        if train:
     

            compo = D.Normal(loc=means, scale=scales)
            compo = D.Independent(compo, 1)
            mix = D.Categorical(logits=logits)
            gmm = D.MixtureSameFamily(
                mixture_distribution=mix, component_distribution=compo
            )
            return gmm
        else:
            max_idx = logits.argmax(dim=-1)
            B, K, O = means.shape
            batch_idx = torch.arange(B)
            chosen_means = means[batch_idx, max_idx]  # (B, output_size)
            chosen_scales = scales[batch_idx, max_idx]
            dist = D.Normal(loc=chosen_means, scale=chosen_scales)
            dist = D.Independent(dist, 1)
            return dist


    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target[:,-1,:])
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

class TemporalGMMHead(nn.Module):
    """
    GMM head producing a MixtureSameFamily distribution over actions.
    Supports deterministic mode extraction for evaluation.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 1024,
        num_layers: int = 2,
        min_std: float = 1e-4,
        num_modes: int = 5,
        low_eval_noise: bool = False,
        loss_coef: float = 1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std
        self.low_eval_noise = low_eval_noise      
        self.loss_coef = loss_coef                

        # Shared MLP
        if num_layers > 0:
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            self.shared_net = nn.Sequential(*layers)
        else:
            self.shared_net = nn.Identity()

        # Output layers
        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

    def forward(self, x: torch.Tensor) -> tD.MixtureSameFamily:
        """
        x: (B, T, input_size)
        returns: GMM distribution of shape (B, T)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        hidden = self.shared_net(x_flat)  # (B*T, hidden_size)

        # Means
        means = self.mean_layer(hidden)  # (B*T, num_modes * output_size)
        means = means.view(B, T, self.num_modes, self.output_size)

        # Log-stds -> positive std
        logstds = self.logstd_layer(hidden)
        logstds = logstds.view(B, T, self.num_modes, self.output_size)
        stds = F.softplus(logstds) + self.min_std

        # Mixing logits
        logits = self.logits_layer(hidden).view(B, T, self.num_modes)

        # Low noise eval: replace std with min_std
        if not self.training and self.low_eval_noise:
            stds = torch.full_like(stds, fill_value=self.min_std)

        # Build distributions
        component_dist = tD.Independent(
            tD.Normal(loc=means, scale=stds), 1
        )  # treats last dim as event dim
        mixture_dist = tD.Categorical(logits=logits)
        gmm = tD.MixtureSameFamily(mixture_dist, component_dist)
        return gmm

    def mode(self, gmm: tD.MixtureSameFamily) -> torch.Tensor:
        """
        Extract the means of the highest-weight Gaussian component.
        returns: (B, T, output_size)
        """
        # logits: (B, T, num_modes)
        logits = gmm.mixture_distribution.logits
        # means: (B, T, num_modes, output_size)
        means = gmm.component_distribution.base_dist.loc
        B, T, M, D = means.shape

        # Index of mode per (B,T)
        mode_idx = logits.argmax(dim=-1)  # (B, T)
        # Gather means
        means_flat = means.view(B * T, M, D)
        idx_flat = mode_idx.view(-1)
        selected = means_flat[torch.arange(B * T, device=means.device), idx_flat]
        return selected.view(B, T, D)

    def loss_fn(self, gmm: tD.MixtureSameFamily, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Negative log-likelihood loss.
        target: (B, T, output_size)
        """
        log_probs = gmm.log_prob(target)
        loss = -log_probs * self.loss_coef
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        if reduction == 'none':
            return loss
        raise ValueError(f"Unknown reduction: {reduction}")
