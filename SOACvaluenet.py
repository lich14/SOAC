import torch
from torch import nn as nn
from SOACutils import LayerNorm, fanin_init


def identity(x):
    return x


class MLP_Net(nn.Module):  # V(S_t)
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.fcs = nn.ModuleList(self.fcs)
        self.layer_norms = nn.ModuleList(self.layer_norms)

        self.last_fc = nn.Linear(in_size, self.output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        return output


class Multitail_Net(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            multi_size,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
            comm_num=0,
    ):
        super().__init__()

        print(f'shared net num is {comm_num}')
        self.input_size = input_size
        self.output_size = output_size
        self.multi_size = multi_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm
        in_size = input_size

        # For simplicity, I force comm_num < 2
        if comm_num == 0:
            self.first_layer = None
        else:
            self.first_layer = nn.Linear(in_size, hidden_sizes[0])
            if self.layer_norm:
                self.first_layernorm = LayerNorm(hidden_sizes[0])
            in_size = hidden_sizes[0]

        self.fcs = [[] for _ in range(multi_size)]
        self.layer_norms = [[] for _ in range(multi_size)]

        for _, next_size in enumerate(hidden_sizes[comm_num:]):
            for j in range(multi_size):
                fc = nn.Linear(in_size, next_size)
                hidden_init(fc.weight)
                fc.bias.data.fill_(b_init_value)
                self.fcs[j].append(fc)

                if self.layer_norm:
                    ln = LayerNorm(next_size)
                    self.layer_norms[j].append(ln)

            in_size = next_size

        self.fcs = nn.ModuleList([nn.ModuleList(fcs) for fcs in self.fcs])
        self.layer_norms = nn.ModuleList(
            [nn.ModuleList(layer_norms) for layer_norms in self.layer_norms])

        self.mean_fcs = []
        self.log_fcs = []

        for i in range(multi_size):
            last_fc = nn.Linear(in_size, self.output_size)
            last_fc.weight.data.uniform_(-init_w, init_w)
            last_fc.bias.data.uniform_(-init_w, init_w)
            self.mean_fcs.append(last_fc)

        for i in range(multi_size):
            last_fc = nn.Linear(in_size, self.output_size)
            last_fc.weight.data.uniform_(-init_w, init_w)
            last_fc.bias.data.uniform_(-init_w, init_w)
            self.log_fcs.append(last_fc)

        self.mean_fcs = nn.ModuleList(self.mean_fcs)
        self.log_fcs = nn.ModuleList(self.log_fcs)

    def forward(self, input):
        h = input
        if self.first_layer:
            h = self.first_layer(h)
            if self.layer_norm:
                h = self.first_layernorm(h)
            h = self.hidden_activation(h)

        means = []
        logs = []

        for tail in range(self.multi_size):
            for i, fc in enumerate(self.fcs[tail]):
                h_cur = fc(h)
                if self.layer_norm and i < len(self.fcs[tail]) - 1:
                    h_cur = self.layer_norms[tail][i](h)
                h_cur = self.hidden_activation(h)

            mean_cur = self.mean_fcs[tail](h_cur)
            mean_cur = self.output_activation(mean_cur)
            means.append(mean_cur)

            log_cur = self.log_fcs[tail](h_cur)
            log_cur = self.output_activation(log_cur)
            logs.append(log_cur)

        mean_out = torch.cat(means, dim=-1)
        log_out = torch.cat(logs, dim=-1)
        return mean_out, log_out


class EmbeddingMlp(MLP_Net):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            option_num,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
    ):
        option_to_dim = input_size
        super().__init__(
            hidden_sizes,
            output_size,
            input_size + option_to_dim,
            init_w,
            hidden_activation,
            hidden_init,
            b_init_value,
            layer_norm,
            output_activation,
        )

        self.embeds = nn.Embedding(option_num, option_to_dim)

    def forward(self, option, *inputs):
        option = self.embeds(option)
        inputs = torch.cat(inputs, dim=1)
        flat_inputs = torch.cat([input, option], dim=1)
        return super().forward(flat_inputs)


class Plus_Net(nn.Module):  # Q(s_t,z_t,a_t)
    def __init__(
            self,
            hidden_sizes,
            output_size,
            state_size,
            action_size,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = state_size + action_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            if i == 0:
                in_size += action_size

            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.fcs = nn.ModuleList(self.fcs)
        self.layer_norms = nn.ModuleList(self.layer_norms)

        self.last_fc = nn.Linear(in_size, self.output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        h = torch.cat((state, action), dim=1)

        for i, fc in enumerate(self.fcs):
            if i == 1:
                h = torch.cat((h, action), dim=1)
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        return output


class PopArt(nn.Module):
    def __init__(self,
                 output_layer,
                 beta: float = 0.0001,
                 zero_debias: bool = False,
                 start_pop: int = 0):
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super(PopArt, self).__init__()
        self.start_pop = start_pop
        self.zero_debias = zero_debias
        self.beta = beta
        self.output_layers = output_layer if isinstance(
            output_layer, (tuple, list, nn.ModuleList)) else (output_layer, )
        #shape = self.output_layers[0].bias.shape
        shape = 1
        device = self.output_layers[0].bias.device
        #assert all(shape == x.bias.shape for x in self.output_layers)
        self.mean = nn.Parameter(torch.zeros(shape, device=device),
                                 requires_grad=False)
        self.mean_square = nn.Parameter(torch.ones(shape, device=device),
                                        requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape, device=device),
                                requires_grad=False)
        self.updates = 0

    def forward(self, *input):
        pass

    @torch.no_grad()
    def update(self, targets):
        targets_shape = targets.shape
        targets = targets.view(-1, 1)
        beta = max(1. / (self.updates + 1.),
                   self.beta) if self.zero_debias else self.beta
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data
        new_mean = (1. - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1. - beta) * self.mean_square + beta * (
            targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(
            0.0001, 1e6)
        assert self.std.shape == (1, ), 'this has only been tested in 1D'
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= self.std / new_std
                layer.bias *= self.std
                layer.bias += self.mean - new_mean
                layer.bias /= new_std
        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.norm(targets).view(*targets_shape)

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, value):
        return value * self.std + self.mean
