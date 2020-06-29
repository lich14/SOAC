import torch
from SOACpolicy import TanhGaussianPolicy
from SOACvaluenet import Plus_Net, MLP_Net
from torch.nn import functional as F


def get_net(obs_dim, action_dim, option_dim, hiddenlayer, layernorm=True):

    qf1 = Plus_Net(
        state_size=obs_dim,
        action_size=action_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    qf1_target = Plus_Net(
        state_size=obs_dim,
        action_size=action_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    qf2 = Plus_Net(
        state_size=obs_dim,
        action_size=action_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    qf2_target = Plus_Net(
        state_size=obs_dim,
        action_size=action_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    u1 = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    u1_target = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    u2 = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    u2_target = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        layer_norm=layernorm,
    )

    policy = TanhGaussianPolicy(
        hiddenlayer,
        obs_dim,
        action_dim,
        option_dim,
        layer_norm=layernorm,
        comm_num=0,
    )

    beta_net = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        output_activation=torch.sigmoid,
        layer_norm=layernorm,
    )

    option_pi_net = MLP_Net(
        input_size=obs_dim,
        output_size=option_dim,
        hidden_sizes=hiddenlayer,
        output_activation=F.softmax,
        layer_norm=layernorm,
    )

    return qf1, qf2, qf1_target, qf2_target, u1, u2, u1_target, u2_target, policy, beta_net, option_pi_net
