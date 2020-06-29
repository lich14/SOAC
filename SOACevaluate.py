import datetime
import numpy as np
import torch
import os
import csv

from SOACbuffer import ReplayBuffer


def getstate(state):
    if isinstance(state, list) is False:
        return state
    return state[0]


def getfirst(list_f):
    if type(list_f[1]) is np.ndarray:
        list_o = []
        for i in list_f:
            list_o.append(i[0])
        return list_o

    else:
        return list_f


class SOACTask():
    def __init__(
            self,
            env,
            trainer,
            obs_dim,
            action_dim,
            option_dim,
            device,
            csv_path,
            start_train_loop=10000,
            test_step_oneloop=5000,
            start_usenet_step=10000,
            batch_size=256,
            buffer_capacity=1000000,
            length=1000000,
            logging=None,
            writer=None,
    ):

        super().__init__()

        self.trainer = trainer
        self.replay_buffer = ReplayBuffer(buffer_capacity, batch_size, obs_dim,
                                          action_dim, option_dim)
        self.env = env
        self.state = getstate(self.env.reset())

        self.logging = logging
        self.device = device
        self.csv_path = csv_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.length = length
        self.writer = writer

        self.test_step_oneloop = test_step_oneloop
        self.is_initial_states = 1

        self.start_train_loop = start_train_loop
        self.start_usenet_step = start_usenet_step
        self.reward = 0
        self.pre_option = 0
        self.episode = 0
        self.episode_steps = 0

        self.pi_z = [0 for i in range(option_dim)]
        self.pi_cur_z = [0 for i in range(option_dim)]
        self.u1_loss, self.u2_loss, self.qf1_loss, self.qf2_loss, self.option_loss = 0, 0, 0, 0, 0
        self.policy_loss, self.alpha, self.IMpara = 0, 0, 0

    def writereward(self, reward, step):
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward])
        else:
            with open(self.csv_path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward'])
                csv_write.writerow([step, reward])

    def to_np(self, a):
        return a.to('cpu').detach().squeeze(0).numpy().tolist()

    def step(self, step):
        if step > self.start_usenet_step:
            option, *_ = self.trainer.select_option(self.state,
                                                    self.pre_option,
                                                    self.is_initial_states,
                                                    False)
            option = self.to_np(option)
            action = self.trainer.select_single_action(self.state, option)
            action = self.to_np(action)
        else:
            option = np.random.randint(self.option_dim)
            action = self.env.action_space.sample()

        self.pi_z[option] += 1
        self.pi_cur_z[option] += 1
        next_state, reward, terminal, info = getfirst(self.env.step(action))
        self.reward += reward
        self.episode_steps += 1
        mask_terminal = False

        if terminal:
            if self.episode_steps >= 999:
                mask_terminal = False
            else:
                mask_terminal = True

        self.replay_buffer.add_sample(self.pre_option, self.state, option,
                                      action, reward, next_state,
                                      int(mask_terminal),
                                      self.is_initial_states)

        self.state = next_state
        self.pre_option = option
        self.is_initial_states = int(terminal)

        if self.replay_buffer.num_transition >= self.start_train_loop and self.replay_buffer.num_transition % 1000 == 999:
            for _ in range(1000):
                B = self.replay_buffer.sample(self.device)
                self.u1_loss, self.u2_loss, self.qf1_loss, self.qf2_loss, self.option_loss, self.policy_loss, self.alpha, self.IMpara, *_ = self.trainer.train_for_list(
                    B,
                    np.array(self.pi_z) / sum(self.pi_z),
                )

        if step % 200000 == 199999:
            self.trainer.save_net(step)

        if terminal:
            self.episode += 1
            print(f'episode: {self.episode:<4}  '
                  f'episode steps: {self.episode_steps:<4}  '
                  f'reward: {self.reward:<5.1f}')

            self.episode_steps = 0
            self.reward = 0
            self.state = getstate(self.env.reset())

        torch.cuda.empty_cache()
        return terminal

    def test(self, step):
        episodes = 10
        returns = np.zeros((episodes, ), dtype=np.float32)
        self.reward = 0

        for i in range(episodes):
            self.state = getstate(self.env.reset())
            episode_reward = 0.
            terminal = False
            while not terminal:
                option = self.trainer.exploit_option(self.state,
                                                     self.pre_option,
                                                     self.is_initial_states)
                option = option.to('cpu').detach().squeeze(0).numpy().tolist()
                action = self.trainer.exploit_action(self.state, option)
                action = action.to('cpu').detach().squeeze(0).numpy().tolist()
                next_state, reward, terminal, info = getfirst(
                    self.env.step(action))
                episode_reward += reward
                self.state = next_state
                self.pre_option = option
                self.is_initial_states = int(terminal)

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        if self.writer:
            self.writer.add_scalar('reward', mean_return, step)
        self.writereward(mean_return, step)

        print(
            "----------------------------------------------------------------------"
        )
        print("Test Steps: {}, Avg. Reward: {}".format(step, mean_return))
        print(f"u1_loss:{self.u1_loss}")
        print(f"u2_loss:{self.u2_loss}")
        print(f"qf1_loss:{self.qf1_loss}")
        print(f"qf2_loss:{self.qf2_loss}")
        print(f"option_loss:{self.option_loss}")
        print(f"policy_loss:{self.policy_loss}")
        print(f"alpha:{self.alpha}")
        print(f"IMpara:{self.IMpara}")
        print(
            f"short option prob:{np.array(self.pi_cur_z) / sum(self.pi_cur_z)}"
        )
        print(f"full option prob:{np.array(self.pi_z) / sum(self.pi_z)}")
        if self.logging:
            self.logging.info(
                f"short option prob:{np.array(self.pi_cur_z) / sum(self.pi_cur_z)}"
            )
            self.logging.info(
                f"full option prob:{np.array(self.pi_z) / sum(self.pi_z)}")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(
            "----------------------------------------------------------------------"
        )
        self.state = getstate(self.env.reset())
        self.pi_cur_z = [0 for i in range(self.option_dim)]
        return mean_return

    def run(self):
        for step in range(self.length):
            self.step(step)
            if step % self.test_step_oneloop == (self.test_step_oneloop - 1):
                reward = self.test(step)
                if self.logging:
                    self.logging.info(
                        f"Step: {step}/{self.length}, Reward: {reward}")
