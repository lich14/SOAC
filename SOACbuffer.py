import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, buffer_capacity, batch_size, obs_dim, action_dim, option_dim):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.preoptions = np.zeros((buffer_capacity))
        self.options = np.zeros((buffer_capacity))

        self.observations = np.zeros((buffer_capacity, obs_dim))
        self.next_obs = np.zeros((buffer_capacity, obs_dim))

        self.actions = np.zeros((buffer_capacity, action_dim))
        self.rewards = np.zeros((buffer_capacity, 1))

        self.terminals = np.zeros((buffer_capacity, 1), dtype='uint8')
        self.initials = np.zeros((buffer_capacity, 1), dtype='uint8')

        self.num_transition = 0

    def add_sample(
            self,
            preoption,
            observation,
            option,
            action,
            reward,
            next_observation,
            terminal,
            initial,
    ):
        if type(observation) is np.ndarray:
            observation = observation.tolist()
        if type(next_observation) is np.ndarray:
            next_observation = next_observation.tolist()

        self.preoptions[self.num_transition % self.buffer_capacity] = preoption
        self.options[self.num_transition % self.buffer_capacity] = option
        self.observations[self.num_transition % self.buffer_capacity] = observation
        self.actions[self.num_transition % self.buffer_capacity] = action
        self.rewards[self.num_transition % self.buffer_capacity] = reward
        self.terminals[self.num_transition % self.buffer_capacity] = terminal
        self.next_obs[self.num_transition % self.buffer_capacity] = next_observation
        self.initials[self.num_transition % self.buffer_capacity] = initial

        self.num_transition += 1

    def sample(self, device):
        index = np.random.randint(0, min(self.num_transition, self.buffer_capacity), (self.batch_size,))

        pre_z = torch.tensor(self.preoptions[index]).reshape(-1, 1).long().to(device)
        s = torch.tensor(self.observations[index]).float().to(device)
        z = torch.tensor(self.options[index]).reshape(-1, 1).long().to(device)
        a = torch.tensor(self.actions[index]).float().to(device)
        r = torch.tensor(self.rewards[index]).reshape(-1, 1).float().to(device)
        s_ = torch.tensor(self.next_obs[index]).float().to(device)
        d = torch.tensor(self.terminals[index]).reshape(-1, 1).float().to(device)
        if_ini = torch.tensor(self.initials[index]).reshape(-1, 1).float().to(device)

        return pre_z, s, z, a, r, s_, d, if_ini
