# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from drqv2 import Actor, Critic, Encoder, RandomShiftsAug, NoShiftAug


class CQLAgent:
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            lr,
            feature_dim,
            hidden_dim,
            critic_target_tau,
            num_expl_steps,
            update_every_steps,
            stddev_schedule,
            stddev_clip,
            use_tb,
            offline=False,
            augmentation=RandomShiftsAug(pad=4),
            # CQL
            cql_importance_sample=False,
            temp=1.0,
            min_q_weight=1.0,
            # sort of backup
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.offline = offline

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = augmentation

        # CQL
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=lr)

        ## min Q
        self.temp = temp
        self.cql_importance_sample = cql_importance_sample
        self.min_q_weight = min_q_weight
        self.num_random = num_random

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        qf1_loss = F.mse_loss(Q1, target_Q)
        qf2_loss = F.mse_loss(Q2, target_Q)

        # add CQL
        if self.offline:
            obs = obs.unsqueeze(1).repeat(1, self.num_random, 1)
            next_obs = next_obs.unsqueeze(1).repeat(1, self.num_random, 1)

            random_actions_tensor = torch.FloatTensor(Q1.shape[0], self.num_random, action.shape[-1]) \
                .uniform_(-1, 1).to(self.device)

            with torch.no_grad():
                curr_dist = self.actor(obs, stddev)
                curr_actions_tensor = curr_dist.sample(clip=self.stddev_clip)
                curr_log_pis = curr_dist.log_prob(curr_actions_tensor).sum(dim=-1, keepdim=True)

                new_curr_dist = self.actor(next_obs, stddev)
                new_curr_actions_tensor = new_curr_dist.sample(clip=self.stddev_clip)
                new_log_pis = new_curr_dist.log_prob(new_curr_actions_tensor).sum(dim=-1, keepdim=True)

            q1_rand, q2_rand = self.critic(obs, random_actions_tensor)
            q1_curr_actions, q2_curr_actions = self.critic(obs, curr_actions_tensor)
            q1_next_actions, q2_next_actions = self.critic(obs, new_curr_actions_tensor)

            if self.cql_importance_sample:
                random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                cat_q1 = torch.cat(
                    [q1_rand - random_density, q1_next_actions - new_log_pis, q1_curr_actions - curr_log_pis], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand - random_density, q2_next_actions - new_log_pis, q2_curr_actions - curr_log_pis], 1
                )
            else:
                cat_q1 = torch.cat([q1_rand, Q1.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
                cat_q2 = torch.cat([q2_rand, Q2.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)

            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - Q1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - Q2.mean() * self.min_q_weight

            if self.with_lagrange:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss

        critic_loss = qf1_loss + qf2_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if self.offline:
                metrics['cql_critic_q1_rand'] = q1_rand.mean().item()
                metrics['cql_critic_q2_rand'] = q2_rand.mean().item()
                metrics['cql_critic_q1_curr_actions'] = q1_curr_actions.mean().item()
                metrics['cql_critic_q2_curr_actions'] = q2_curr_actions.mean().item()
                metrics['cql_critic_q1_next_actions'] = q1_next_actions.mean().item()
                metrics['cql_critic_q2_next_actions'] = q2_next_actions.mean().item()
                metrics['cql_critic_q1_loss'] = min_qf1_loss.item()
                metrics['cql_critic_q2_loss'] = min_qf2_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = -Q.mean()
        actor_loss = actor_policy_improvement_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
