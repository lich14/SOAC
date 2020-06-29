import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import SOACutils as ptu
from SOACvaluenet import PopArt
import datetime


class SOACTrainer():
    def __init__(
            self,
            policy,
            q1,
            q2,
            q1_target,
            q2_target,
            u1,
            u2,
            u1_target,
            u2_target,
            beta_net,
            option_pi_net,
            device,
            option_dim,
            state_dim,
            logger,
            lr=3e-4,
            ifpopart=True,
            IMpara=0.1,
            target_alpha=1.,
            load_path=None,
            discount=0.99,
            reward_scale=5.0,
            soft_target_tau=5e-3,
            target_update_period=1,
            optimizer_class=optim.Adam,
            anneal=False,
            length=1e6,
            writer=None,
    ):

        super().__init__()
        """
        =================================================================================
        Net Group
        =================================================================================
        """
        self.q1 = q1.to(device)
        self.q2 = q2.to(device)
        self.q1_target = q1_target.to(device)
        self.q2_target = q2_target.to(device)

        self.u1 = u1.to(device)
        self.u2 = u2.to(device)
        self.u1_target = u1_target.to(device)
        self.u2_target = u2_target.to(device)

        self.policy = policy.to(device)
        self.beta_net = beta_net.to(device)
        self.option_pi_net = option_pi_net.to(device)

        ptu.update_from_to(self.q1, self.q1_target)
        ptu.update_from_to(self.q2, self.q2_target)
        ptu.update_from_to(self.u1, self.u1_target)
        ptu.update_from_to(self.u2, self.u2_target)

        if ifpopart:
            self.q1_normer = PopArt(self.q1.last_fc)
            self.q2_normer = PopArt(self.q2.last_fc)
            self.q1_target_normer = PopArt(self.q1_target.last_fc)
            self.q2_target_normer = PopArt(self.q2_target.last_fc)
            self.u1_normer = PopArt(self.u1.last_fc)
            self.u2_normer = PopArt(self.u2.last_fc)
            self.u1_target_normer = PopArt(self.u1_target.last_fc)
            self.u2_target_normer = PopArt(self.u2_target.last_fc)
        """
        =================================================================================
        Env param
        =================================================================================
        """
        self.state_dim = state_dim
        self.option_dim = option_dim
        self.logger = logger
        self.device = device
        self.load_path = load_path
        self.writer = writer

        self.mean_lambda = 1e-3
        self.std_lambda = 1e-3
        self.discount = discount
        self.reward_scale = reward_scale
        self.n_train_steps_total = 0
        self.target_update_period = target_update_period
        """
        =================================================================================
        Important param
        =================================================================================
        """
        self.soft_target_tau = soft_target_tau
        self.ifaddconstrain = True

        if ifpopart:
            self.critic_clip = True
            self.actor_clip = True
            self.critic_grad = 100.
            self.actor_grad = 100.
            self.ifpopart = True

        else:
            self.critic_clip = False
            self.actor_clip = True
            self.actor_grad = 10.
            self.ifpopart = False

        self.ifdouble_u = True
        self.ifdouble_q = True
        self.IMpara = IMpara
        self.target_alpha = target_alpha
        """
        =================================================================================
        Training param
        =================================================================================
        """
        self.criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.q1_optimizer = optimizer_class(
            self.q1.parameters(),
            lr=lr,
        )
        self.q2_optimizer = optimizer_class(
            self.q2.parameters(),
            lr=lr,
        )

        self.u1_optimizer = optimizer_class(
            self.u1.parameters(),
            lr=lr,
        )
        self.u2_optimizer = optimizer_class(
            self.u2.parameters(),
            lr=lr,
        )

        self.beta_optimizer = optimizer_class(
            self.beta_net.parameters(),
            lr=lr,
        )
        self.option_pi_optimizer = optimizer_class(
            self.option_pi_net.parameters(),
            lr=lr,
        )

        self.anneal = anneal
        if anneal:
            self.load_anneal_lr(T_max=length)

    def load_anneal_lr(self, T_max=1e6, lr_min=1e-6):
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=T_max, eta_min=lr_min)
        self.q1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q1_optimizer, T_max=T_max, eta_min=lr_min)
        self.q2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q2_optimizer, T_max=T_max, eta_min=lr_min)
        self.u1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.u1_optimizer, T_max=T_max, eta_min=lr_min)
        self.u2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.u2_optimizer, T_max=T_max, eta_min=lr_min)
        self.beta_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.beta_optimizer, T_max=T_max, eta_min=lr_min)
        self.option_pi_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.option_pi_optimizer, T_max=T_max, eta_min=lr_min)

    def save_net(self, step):
        state = {
            'q1_net': self.q1.state_dict(),
            'q2_net': self.q2.state_dict(),
            'u1_net': self.u1.state_dict(),
            'u2_net': self.u2.state_dict(),
            'policy_net': self.policy.state_dict(),
            'beta_net': self.beta_net.state_dict(),
            'option_pi_net': self.option_pi_net.state_dict(),
        }

        torch.save(state, self.load_path + 'SOAC' + str(step) + '.lch')

    def load_net(self, step):
        param = torch.load(self.load_path + 'SOAC' + str(step) + '.lch')
        self.q1.load_state_dict(param['q1_net'])
        self.q2.load_state_dict(param['q2_net'])
        self.q1_target.load_state_dict(param['q1_net'])
        self.q2_target.load_state_dict(param['q2_net'])

        self.u1.load_state_dict(param['u1_net'])
        self.u2.load_state_dict(param['u2_net'])
        self.u1_target.load_state_dict(param['u1_net'])
        self.u2_target.load_state_dict(param['u2_net'])

        self.policy.load_state_dict(param['policy_net'])
        self.beta_net.load_state_dict(param['beta_net'])
        self.option_pi_net.load_state_dict(param['option_pi_net'])
        print('load step:', step)

    def select_option(
            self,
            state,
            pre_option,
            ifinitial,
            return_log_prob=True,
    ):

        if np.shape(ifinitial) == ():
            state = torch.tensor(state).float().to(self.device).view(
                -1, self.state_dim)
            pre_option = torch.tensor(pre_option).long().to(self.device).view(
                -1, 1)
            ifinitial = torch.tensor(ifinitial).long().to(self.device).view(
                -1, 1)  # 1:initial 0:no

        beta = self.beta_net(state)
        beta_withpreoption = torch.gather(beta, 1, pre_option)

        q = self.option_pi_net(state)
        q_rechoose = q.clone().scatter_(1, pre_option.long(), -1e20)

        mask = torch.zeros_like(q).scatter_(1, pre_option.long(),
                                            torch.ones_like(q))
        q_rechoose_softmax = torch.softmax(q_rechoose, dim=-1)
        pi = ifinitial * q + (1 - ifinitial) * (
            (1 - beta_withpreoption) * mask +
            beta_withpreoption * q_rechoose_softmax)

        dist = torch.distributions.Categorical(probs=pi)
        option = dist.sample()
        option_logp = None
        if return_log_prob:
            option_logp = dist.log_prob(option)

        return option.long(), option_logp, pi, beta, q

    def exploit_option(
            self,
            state,
            pre_option,
            ifinitial,
    ):

        if np.shape(ifinitial) == ():
            state = torch.tensor(state).float().to(self.device).view(
                -1, self.state_dim)
            pre_option = torch.tensor(pre_option).long().to(self.device).view(
                -1, 1)
            ifinitial = torch.tensor(ifinitial).long().to(self.device).view(
                -1, 1)  # 1:initial 0:no

        beta = self.beta_net(state)
        beta_withpreoption = torch.gather(beta, 1, pre_option)

        q = self.option_pi_net(state)
        q_rechoose = q.clone().scatter_(1, pre_option.long(), -1e20)

        mask = torch.zeros_like(q).scatter_(1, pre_option.long(),
                                            torch.ones_like(q))
        q_rechoose_softmax = torch.softmax(q_rechoose, dim=-1)
        pi = ifinitial * q + (1 - ifinitial) * (
            (1 - beta_withpreoption) * mask +
            beta_withpreoption * q_rechoose_softmax)

        option = torch.argmax(pi, dim=-1)

        return option.long()

    def select_single_action(self, state, option):
        if np.shape(option) == ():
            state = torch.tensor(state).float().to(self.device).view(
                -1, self.state_dim)
            option = torch.tensor(option).long().to(self.device).view(-1, 1)

        new_actions, *_ = self.policy(
            state,
            option,
            reparameterize=True,
            return_log_prob=False,
        )

        return new_actions

    def exploit_action(self, state, option):
        if np.shape(option) == ():
            state = torch.tensor(state).float().to(self.device).view(
                -1, self.state_dim)
            option = torch.tensor(option).long().to(self.device).view(-1, 1)

        new_actions, *_ = self.policy(
            state,
            option,
            reparameterize=True,
            return_log_prob=False,
            deterministic=True,
        )

        return new_actions

    def evaluate_action(self, state, action):  # got p(a|s,z) for all z
        log_prob_array = self.policy.get_option_logp(state, action).exp()
        return log_prob_array

    def train_for_list(self, B, pi_z):
        if self.n_train_steps_total % 500 == 0:
            print("Training ... {} times ".format(self.n_train_steps_total))
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(
                "----------------------------------------------------------------------"
            )

        pre_z, s, z, a, r, s_, d, if_ini = B
        alpha = torch.tensor(self.target_alpha).type_as(s)
        """
        =================================================================================
        Basic
        =================================================================================
        """
        new_option, new_option_logp, pi, *_ = self.select_option(
            s, pre_z, if_ini, return_log_prob=True)  # sample a single option
        new_option, new_option_logp = new_option.view(-1,
                                                      1), new_option_logp.view(
                                                          -1, 1)
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            s,
            new_option,
            reparameterize=True,
            return_log_prob=True,
        )
        """
        =================================================================================
        Q Loss and U loss
        =================================================================================
        """
        new_option_next, new_option_logp_next, pi_next, *_ = self.select_option(
            s_, z, d, True)
        new_option_next, new_option_logp_next = new_option_next.view(
            -1, 1), new_option_logp_next.view(-1, 1)

        with torch.no_grad():
            u1_next_state_target = torch.gather(self.u1_target(s_), 1,
                                                new_option_next)
            if self.ifdouble_u:
                u2_next_state_target = torch.gather(self.u2_target(s_), 1,
                                                    new_option_next)

            action_pi = self.evaluate_action(s, a)
            matrix_pi = (pi * action_pi).clamp(1e-10)
            sum_pi = (matrix_pi).sum(dim=-1, keepdim=True)  # p(a|s)
            each_pi = (matrix_pi / sum_pi).clamp(1e-10)

            p_z = each_pi.sum(dim=0)
            p_z = p_z / p_z.sum()

            bn_mask = torch.zeros_like(pi).scatter(1, z, 1)
            cur_pi = (each_pi * bn_mask).sum(dim=-1, keepdim=True).clamp(
                1e-10)  # p(z|s,a) z is the original option

            H_o = torch.tensor([
                -p_z[i] * torch.log(p_z[i].clamp(1e-10))
                for i in range(self.option_dim)
            ]).sum()
            H_o_sa = (-cur_pi *
                      torch.log(cur_pi.clamp(1e-10))).sum() / cur_pi.shape[0]
            IM = H_o - H_o_sa

            s_c = s + torch.tensor(np.random.normal(
                0, 1, s.shape)).type_as(s).clamp(-1, 1)
            a_c = (a + torch.tensor(np.random.normal(
                0, 0.2, a.shape)).type_as(a).clamp(-0.2, 0.2)).clamp(
                    -1 + 1e-5, 1 - 1e-5)
            action_pi_c = self.evaluate_action(s_c, a_c)
            _, _, pi_c, *_ = self.select_option(
                s_c, pre_z, if_ini,
                return_log_prob=True)  # sample a single option
            matrix_pi_c = (pi_c * action_pi_c).clamp(1e-10)
            sum_pi_c = matrix_pi_c.sum(dim=-1, keepdim=True).clamp(1e-10)
            each_pi_c = (matrix_pi_c / sum_pi_c).clamp(1e-10)

            cur_pi_c = (each_pi_c * bn_mask).sum(dim=-1, keepdim=True).clamp(
                1e-10)  # p(z|s,a)
            TV = torch.abs(cur_pi - cur_pi_c)
            bias_noise = TV
            add_item = alpha * (self.IMpara * IM - 5 * bias_noise -
                                torch.log(cur_pi))

            if self.ifpopart:
                u1_next_state_target_unnorm = self.u1_target_normer.unnorm(
                    u1_next_state_target)
                if self.ifdouble_u:
                    u2_next_state_target_unnorm = self.u2_target_normer.unnorm(
                        u2_next_state_target)
                    u_next_state_target_unnorm = torch.min(
                        u1_next_state_target_unnorm,
                        u2_next_state_target_unnorm)
                else:
                    u_next_state_target_unnorm = u1_next_state_target_unnorm

                q_target_value_unnorm = u_next_state_target_unnorm - alpha * new_option_logp_next
                q1_target_unnorm = self.reward_scale * r + add_item + (
                    1. - d) * self.discount * q_target_value_unnorm
                if self.ifdouble_q:
                    q2_target_unnorm = q1_target_unnorm

                q1_target = self.q1_normer.update(q1_target_unnorm)
                if self.ifdouble_q:
                    q2_target = self.q2_normer.update(q2_target_unnorm)

            else:
                if self.ifdouble_u:
                    u_next_state_target = torch.min(u1_next_state_target,
                                                    u2_next_state_target)
                else:
                    u_next_state_target = u1_next_state_target

                q_target_value = u_next_state_target - alpha * new_option_logp_next
                q1_target = self.reward_scale * r + add_item + (
                    1. - d) * self.discount * q_target_value
                if self.ifdouble_q:
                    q2_target = q1_target

        q1_pred = self.q1(s, a)
        q1_pred_withoption = torch.gather(q1_pred, 1, z)
        q1_loss = self.criterion(q1_pred_withoption, q1_target.detach())

        if self.ifdouble_q:
            q2_pred = self.q2(s, a)
            q2_pred_withoption = torch.gather(q2_pred, 1, z)
            q2_loss = self.criterion(q2_pred_withoption, q2_target.detach())
        else:
            q2_loss = None

        new_actions_oldoption, _, _, new_log_pi_oldoption, *_ = self.policy(
            s,
            z,
            reparameterize=True,
            return_log_prob=True,
        )

        with torch.no_grad():
            q1_cal_u = self.q1_target(s, new_actions_oldoption)
            q1_cal_u_withoption = torch.gather(q1_cal_u, 1, z)

            if self.ifdouble_q:
                q2_cal_u = self.q2_target(s, new_actions_oldoption)
                q2_cal_u_withoption = torch.gather(q2_cal_u, 1, z)

            if self.ifpopart:
                q1_cal_u_withoption_unnorm = self.q1_target_normer.unnorm(
                    q1_cal_u_withoption)
                if self.ifdouble_q:
                    q2_cal_u_withoption_unnorm = self.q2_target_normer.unnorm(
                        q2_cal_u_withoption)
                    q_cal_u_unnorm = torch.min(q1_cal_u_withoption_unnorm,
                                               q2_cal_u_withoption_unnorm)
                else:
                    q_cal_u_unnorm = q1_cal_u_withoption_unnorm

                u_target_value_unnorm = q_cal_u_unnorm - alpha * new_log_pi_oldoption

                u1_target_value = self.u1_normer.update(u_target_value_unnorm)
                if self.ifdouble_u:
                    u2_target_value = self.u2_normer.update(
                        u_target_value_unnorm)
            else:
                if self.ifdouble_q:
                    q_cal_u = torch.min(q1_cal_u_withoption,
                                        q2_cal_u_withoption)
                else:
                    q_cal_u = q1_cal_u_withoption

                u1_target_value = q_cal_u - alpha * new_log_pi_oldoption
                if self.ifdouble_u:
                    u2_target_value = q_cal_u - alpha * new_log_pi_oldoption

        u1_pred = self.u1(s)
        u1_pred_withoption = torch.gather(u1_pred, 1, z)
        u1_loss = self.criterion(u1_pred_withoption, u1_target_value)

        if self.ifdouble_u:
            u2_pred = self.u2(s)
            u2_pred_withoption = torch.gather(u2_pred, 1, z)
            u2_loss = self.criterion(u2_pred_withoption, u2_target_value)
        else:
            u2_loss = None
        """
        =================================================================================
        policy loss & option loss
        =================================================================================
        """
        q1_action = self.q1(s, new_actions)
        q1_action_withoption = torch.gather(q1_action, 1, new_option)

        if self.ifdouble_q:
            q2_action = self.q2(s, new_actions)
            q2_action_withoption = torch.gather(q2_action, 1, new_option)

        if self.ifpopart:
            q1_action_withoption_unnorm = self.q1_normer.unnorm(
                q1_action_withoption)
            if self.ifdouble_q:
                q2_action_withoption_unnorm = self.q2_normer.unnorm(
                    q2_action_withoption)
                q_action = torch.min(q1_action_withoption_unnorm,
                                     q2_action_withoption_unnorm)
            else:
                q_action = q1_action_withoption_unnorm

        else:
            if self.ifdouble_q:
                q_action = torch.min(q1_action_withoption,
                                     q2_action_withoption)
            else:
                q_action = q1_action_withoption

        policy_loss = (alpha * log_pi - q_action).mean()
        if self.ifaddconstrain:
            mean_loss = self.mean_lambda * policy_mean.pow(2).mean()
            std_loss = self.std_lambda * policy_log_std.pow(2).mean()
            policy_loss = policy_loss + mean_loss + std_loss

        if self.ifpopart:
            if self.ifdouble_q:
                policy_loss = 0.5 * (self.q1_normer.norm(policy_loss) +
                                     self.q2_normer.norm(policy_loss))
            else:
                policy_loss = self.q1_normer.norm(policy_loss)

        u1_new = self.u1(s)
        if self.ifdouble_u:
            u2_new = self.u2(s)

        if self.ifpopart:
            u1_new = self.u1_normer.unnorm(u1_new)
            if self.ifdouble_u:
                u2_new = self.u2_normer.unnorm(u2_new)

        if self.ifdouble_u:
            u_new = torch.min(u1_new, u2_new)
        else:
            u_new = u1_new

        minus = alpha * torch.log(pi.clamp(1e-10)) - u_new
        option_loss = (pi * minus).sum(dim=-1, keepdim=True).mean()

        if self.ifpopart:
            if self.ifdouble_u:
                option_loss = 0.5 * (self.u1_normer.norm(option_loss) +
                                     self.u2_normer.norm(option_loss))
            else:
                option_loss = self.u1_normer.norm(option_loss)
        """
        =================================================================================
        Update networks
        =================================================================================
        """
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.critic_clip:
            nn.utils.clip_grad_norm_(self.q1.parameters(), self.critic_grad)
        self.q1_optimizer.step()

        if self.ifdouble_q:
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            if self.critic_clip:
                nn.utils.clip_grad_norm_(self.q2.parameters(),
                                         self.critic_grad)
            self.q2_optimizer.step()

        self.u1_optimizer.zero_grad()
        u1_loss.backward()
        if self.critic_clip:
            nn.utils.clip_grad_norm_(self.u1.parameters(), self.critic_grad)
        self.u1_optimizer.step()

        if self.ifdouble_u:
            self.u2_optimizer.zero_grad()
            u2_loss.backward()
            if self.critic_clip:
                nn.utils.clip_grad_norm_(self.u2.parameters(),
                                         self.critic_grad)
            self.u2_optimizer.step()

        self.beta_optimizer.zero_grad()
        option_loss.backward(retain_graph=True)
        if self.actor_clip:
            nn.utils.clip_grad_norm_(self.beta_net.parameters(),
                                     self.actor_grad)
        self.beta_optimizer.step()

        self.option_pi_optimizer.zero_grad()
        option_loss.backward(retain_graph=True)
        if self.actor_clip:
            nn.utils.clip_grad_norm_(self.option_pi_net.parameters(),
                                     self.actor_grad)
        self.option_pi_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.actor_clip:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.actor_grad)
        self.policy_optimizer.step()

        if self.anneal:
            self.policy_scheduler.step()
            self.q1_scheduler.step()
            self.q2_scheduler.step()
            self.u1_scheduler.step()
            self.u2_scheduler.step()
            self.beta_scheduler.step()
            self.option_pi_scheduler.step()
        """
        =================================================================================
        Soft Updates
        =================================================================================
        """
        if self.n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.q1, self.q1_target,
                                    self.soft_target_tau)
            if self.ifdouble_q:
                ptu.soft_update_from_to(self.q2, self.q2_target,
                                        self.soft_target_tau)

            ptu.soft_update_from_to(self.u1, self.u1_target,
                                    self.soft_target_tau)
            if self.ifdouble_u:
                ptu.soft_update_from_to(self.u2, self.u2_target,
                                        self.soft_target_tau)

            if self.ifpopart:
                ptu.soft_update_from_to(self.q1_normer, self.q1_target_normer,
                                        self.soft_target_tau)
                if self.ifdouble_q:
                    ptu.soft_update_from_to(self.q2_normer,
                                            self.q2_target_normer,
                                            self.soft_target_tau)

                ptu.soft_update_from_to(self.u1_normer, self.u1_target_normer,
                                        self.soft_target_tau)
                if self.ifdouble_u:
                    ptu.soft_update_from_to(self.u2_normer,
                                            self.u2_target_normer,
                                            self.soft_target_tau)

        self.n_train_steps_total += 1
        if self.writer and self.n_train_steps_total % 10 == 9:
            self.writer.add_scalar('loss/q1_loss', q1_loss,
                                   self.n_train_steps_total)
            self.writer.add_scalar('loss/q2_loss', q2_loss,
                                   self.n_train_steps_total)
            self.writer.add_scalar('loss/u1_loss', u1_loss,
                                   self.n_train_steps_total)
            self.writer.add_scalar('loss/u2_loss', u2_loss,
                                   self.n_train_steps_total)
            self.writer.add_scalar('loss/policy_loss', policy_loss,
                                   self.n_train_steps_total)
            self.writer.add_scalar('loss/option_loss', option_loss,
                                   self.n_train_steps_total)

        return u1_loss, u2_loss, q1_loss, q2_loss, option_loss, policy_loss, alpha, self.IMpara
