import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from SOACdistribution import TanhNormal
from SOACvaluenet import MLP_Net, Multitail_Net
import time

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Multitail_Net):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            option_num,
            init_w=1e-3,
            hidden_activation=torch.relu,
            layer_norm=True,
            comm_num=0,
            **kwargs,
    ):

        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            multi_size=option_num,
            init_w=init_w,
            hidden_activation=hidden_activation,
            layer_norm=layer_norm,
            comm_num=0,
            **kwargs,
        )

        self.option_dim = option_num
        self.action_dim = action_dim

        self.array_choose = torch.arange(0, action_dim * option_num).view(
            -1, action_dim)

    def forward(
            self,
            obs,
            option=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        if option is not None:
            if option.shape[0] > 1:
                stack_a = option * self.action_dim
                option_array = torch.cat(
                    [stack_a + i for i in range(self.action_dim)],
                    dim=-1).type_as(obs).long()

            else:
                option_array = self.array_choose[option[0]].view(
                    -1, self.action_dim).type_as(obs).long()

        mean, logs = super().forward(obs)
        if option is not None:
            mean = torch.gather(mean, 1, option_array)

        log_std = torch.clamp(logs, LOG_SIG_MIN, LOG_SIG_MAX)
        if option is not None:
            log_std = torch.gather(log_std, 1, option_array)

        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True)

                log_prob = tanh_normal.log_prob(action,
                                                pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()

                else:
                    action = tanh_normal.sample()

        return (
            action,
            mean,
            log_std,
            log_prob,
            entropy,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    def get_logp(self, state, option, action, epsilon=1e-6):
        _, mean, _, _, _, std, *_ = self.forward(state, option)
        action = action.clamp(-1 + 1e-5, 1 - 1e-5)
        u = 0.5 * torch.log((action + 1) / (1 - action))
        log_prob = Normal(
            mean, std).log_prob(u) - torch.log(1 - action * action + epsilon)
        log_prob_p = log_prob.sum(dim=1, keepdim=True)

        return log_prob_p

    def get_option_logp(self, state, action, epsilon=1e-6):
        _, mean, _, _, _, std, *_ = self.forward(state)
        action = action.clamp(-1 + 1e-5, 1 - 1e-5)
        u = 0.5 * torch.log((action + 1) / (1 - action))
        log_prob = torch.cat([
            (Normal(mean[:, i * self.action_dim:(i + 1) * self.action_dim],
                    std[:, i * self.action_dim:(i + 1) *
                        self.action_dim]).log_prob(u) -
             torch.log(1 - action * action + epsilon)).sum(dim=1, keepdim=True)
            for i in range(self.option_dim)
        ],
                             dim=-1)

        return log_prob
