import argparse
from copy import deepcopy
import functools
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

from baselines.lompo import buffer
from baselines.lompo import models
from baselines.lompo import tools
from baselines.lompo import wrappers


def define_config():
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('.logdir')
    config.loaddir = pathlib.Path('.logdir')
    config.datadir = pathlib.Path('.datadir/walker')
    config.seed = 0
    config.log_every = 1000
    config.save_every = 5000
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True

    # Environment.
    config.task = 'dmc_walker_walk'
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 2
    config.time_limit = 1000
    config.im_size = 64
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    config.precision = 32

    # Model.
    config.deter_size = 256
    config.stoch_size = 64
    config.num_models = 7
    config.num_units = 256
    config.proprio = False
    config.penalty_type = 'log_prob'
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'

    # Training.
    config.load_model = False
    config.load_agent = False
    config.load_buffer = False
    config.train_steps = 100000
    config.model_train_steps = 25000
    config.model_batch_size = 64
    config.model_batch_length = 50
    config.agent_batch_size = 256
    config.cql_samples = 16
    config.start_training = 50000
    config.agent_train_steps = 150000
    config.agent_iters_per_step = 200
    config.buffer_size = 2e6
    config.model_lr = 6e-4
    config.q_lr = 3e-4
    config.actor_lr = 3e-4
    config.grad_clip = 100.0
    config.tau = 5e-3
    config.target_update_interval = 1
    config.dataset_balance = False

    # Behavior.
    config.lmbd = 5.0
    config.alpha = 0.0
    config.sample = True
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 5
    config.done_treshold = 0.5
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.2
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config


class Lompo(tools.Module):

    def __init__(self, config, datadir, actspace, writer):
        self._c = config
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
        episodes, steps = tools.count_episodes(datadir)
        self.latent_buffer = buffer.LatentReplayBuffer(steps,
                                                       steps,
                                                       self._c.deter_size + self._c.stoch_size,
                                                       self._actdim)
        self.lmbd = config.lmbd
        self.alpha = config.alpha

        self._writer = writer
        tf.summary.experimental.set_step(0)
        self._metrics = dict()

        self._agent_step = 0
        self._model_step = 0

        self._random = np.random.RandomState(config.seed)
        self._float = prec.global_policy().compute_dtype
        self._dataset = iter(load_dataset(datadir, self._c))
        self._episode_iterator = episode_iterator(datadir, self._c)

        self._build_model()
        for _ in range(10):
            self._model_train_step(next(self._dataset), prefix='eval')

    def __call__(self, obs, reset, state=None, training=True):
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        action, state = self.policy(obs, state, training)
        return action, state

    def load(self, filename):
        try:
            self.load_model(filename)
        except:
            pass

        try:
            self.load_agent(filename)
        except:
            pass

    def save(self, filename):
        self.save_model(filename)
        self.save_agentl(filename)

    def load_model(self, filename):
        self._encode.load(filename / 'encode.pkl')
        self._dynamics.load(filename / 'dynamic.pkl')
        self._decode.load(filename / 'decode.pkl')
        self._reward.load(filename / 'reward.pkl')
        if self._c.pcont:
            self._pcont.load(filename / 'pcont.pkl')
        if self._c.proprio:
            self._proprio.load(filename / 'proprio.pkl')

    def save_model(self, filename):
        filename.mkdir(parents=True, exist_ok=True)
        self._encode.save(filename / 'encode.pkl')
        self._dynamics.save(filename / 'dynamic.pkl')
        self._decode.save(filename / 'decode.pkl')
        self._reward.save(filename / 'reward.pkl')
        if self._c.pcont:
            self._pcont.save(filename / 'pcont.pkl')
        if self._c.proprio:
            self._proprio.save(filename / 'proprio.pkl')

    def load_agent(self, filename):
        self._qf1.load(filename / 'qf1.pkl')
        self._qf2.load(filename / 'qf2.pkl')
        self._target_qf1.load(filename / 'target_qf1.pkl')
        self._target_qf2.load(filename / 'target_qf2.pkl')
        self._actor.load(filename / 'actor.pkl')

    def save_agent(self, filename):
        filename.mkdir(parents=True, exist_ok=True)
        self._qf1.save(filename / 'qf1.pkl')
        self._qf2.save(filename / 'qf2.pkl')
        self._target_qf1.save(filename / 'target_qf1.pkl')
        self._target_qf2.save(filename / 'target_qf2.pkl')
        self._actor.save(filename / 'actor.pkl')

    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float)
        else:
            latent, action = state

        embed = self._encode(preprocess_raw(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)
        action = self._exploration(self._actor(feat), training)
        state = (latent, action)
        return action, state

    def _build_model(self):
        acts = dict(elu=tf.nn.elu, relu=tf.nn.relu,
                    swish=tf.nn.swish, leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]

        # Create encoder based on environment observations
        if self._c.proprio:
            if self._c.im_size == 64:
                self._encode = models.ConvEncoderProprio(self._c.cnn_depth, cnn_act)
            else:
                self._encode = models.ConvEncoderProprioLarge(self._c.cnn_depth, cnn_act)
        else:
            if self._c.im_size == 64:
                self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
            else:
                self._encode = models.ConvEncoderLarge(self._c.cnn_depth, cnn_act)
                # RSSM model with ensembles
        self._dynamics = models.RSSME(self._c.stoch_size, self._c.deter_size,
                                      self._c.deter_size, num_models=self._c.num_models)
        # Create decoder based on image size
        if self._c.im_size == 64:
            self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act,
                                              shape=(self._c.im_size, self._c.im_size, 3))
        else:
            self._decode = models.ConvDecoderLarge(self._c.cnn_depth, cnn_act,
                                                   shape=(self._c.im_size, self._c.im_size, 3))
        if self._c.proprio:
            self._proprio = models.DenseDecoder((self._propriodim,), 3, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder((), 3, self._c.num_units, 'binary', act=act)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)

        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self._c.proprio:
            model_modules.append(self._proprio)
        if self._c.pcont:
            model_modules.append(self._pcont)

        # Build actor-critic networks
        self._qf1 = models.DenseNetwork(1, 3, self._c.num_units, act=act)
        self._qf2 = models.DenseNetwork(1, 3, self._c.num_units, act=act)
        self._target_qf1 = deepcopy(self._qf2)
        self._target_qf2 = deepcopy(self._qf1)
        self._qf_criterion = tf.keras.losses.Huber()
        self._actor = models.ActorNetwork(self._actdim, 4, self._c.num_units, act=act)

        # Initialize optimizers
        Optimizer = functools.partial(tools.Adam,
                                      wd=self._c.weight_decay,
                                      clip=self._c.grad_clip,
                                      wdpattern=self._c.weight_decay_pattern)

        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._qf_opt = Optimizer('qf', [self._qf1, self._qf2], self._c.q_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._agent_step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'] = amount
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        raise NotImplementedError(self._c.expl)

    def fit_model(self, iters):
        for iter in range(iters):
            data = next(self._dataset)
            self._model_train_step(data)
            if iter % self._c.save_every == 0:
                self.save_model(self._c.logdir / 'model_step_{}'.format(iter))
        self.save_model(self._c.logdir / 'final_model')

    def _model_train_step(self, data, prefix='train'):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)
            image_pred = self._decode(feat)
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()
            likes.image = tf.reduce_mean(tf.boolean_mask(image_pred.log_prob(data['image']),
                                                         data['mask']))
            likes.reward = tf.reduce_mean(tf.boolean_mask(reward_pred.log_prob(data['reward']),
                                                          data['mask']))
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = data['terminal']
                likes.pcont = tf.reduce_mean(tf.boolean_mask(pcont_pred.log_prob(pcont_target),
                                                             data['mask']))
                likes.pcont *= self._c.pcont_scale

            for key in prior.keys():
                prior[key] = tf.boolean_mask(prior[key], data['mask'])
                post[key] = tf.boolean_mask(post[key], data['mask'])

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            model_loss = self._c.kl_scale * div - sum(likes.values())

        if prefix == 'train':
            model_norm = self._model_opt(model_tape, model_loss)
            self._model_step += 1

        if self._model_step % self._c.log_every == 0:
            self._image_summaries(data, embed, image_pred, self._model_step, prefix)
            model_summaries = dict()
            model_summaries['model_train/KL Divergence'] = tf.reduce_mean(div)
            model_summaries['model_train/image_recon'] = tf.reduce_mean(likes.image)
            model_summaries['model_train/reward_recon'] = tf.reduce_mean(likes.reward)
            model_summaries['model_train/model_loss'] = tf.reduce_mean(model_loss)
            if prefix == 'train':
                model_summaries['model_train/model_norm'] = tf.reduce_mean(model_norm)
            if self._c.pcont:
                model_summaries['model_train/terminal_recon'] = tf.reduce_mean(likes.pcont)
            self._write_summaries(model_summaries, self._model_step)

    def train_agent(self, iters):
        for iter in range(iters):
            data = preprocess_latent(self.latent_buffer.sample(self._c.agent_batch_size))
            self._agent_train_step(data)
            if self._agent_step % self._c.target_update_interval == 0:
                self._update_target_critics()
            if iter % self._c.save_every == 0:
                self.save_agent(self._c.logdir)
        self.save_agent(self._c.logdir / 'final_agent')

    def _agent_train_step(self, data):
        obs = data['obs']
        actions = data['actions']
        next_obs = data['next_obs']
        rewards = data['rewards']
        terminals = data['terminals']

        with tf.GradientTape() as q_tape:
            q1_pred = self._qf1(tf.concat([obs, actions], axis=-1))
            q2_pred = self._qf2(tf.concat([obs, actions], axis=-1))
            # new_next_actions = self._exploration(self._actor(next_obs), True)
            new_actions = self._actor(obs)
            new_next_actions = self._actor(next_obs)

            target_q_values = tf.reduce_min([self._target_qf1(tf.concat([next_obs, new_next_actions], axis=-1)),
                                             self._target_qf2(tf.concat([next_obs, new_next_actions], axis=-1))],
                                            axis=0)
            q_target = rewards + self._c.discount * (1.0 - terminals) * target_q_values

            expanded_actions = tf.expand_dims(actions, 0)
            tilled_actions = tf.tile(expanded_actions, [self._c.cql_samples, 1, 1])
            tilled_actions = tf.random.uniform(tilled_actions.shape, minval=-1, maxval=1)
            tilled_actions = tf.concat([tilled_actions, tf.expand_dims(new_actions, 0)], axis=0)

            expanded_obs = tf.expand_dims(obs, 0)
            tilled_obs = tf.tile(expanded_obs, [self._c.cql_samples + 1, 1, 1])

            q1_values = self._qf1(tf.concat([tilled_obs, tilled_actions], axis=-1))
            q2_values = self._qf2(tf.concat([tilled_obs, tilled_actions], axis=-1))
            q1_penalty = tf.math.reduce_logsumexp(q1_values, axis=0)
            q2_penalty = tf.math.reduce_logsumexp(q2_values, axis=0)

            qf1_loss = self.alpha * (tf.reduce_mean(q1_penalty) - tf.reduce_mean(q1_pred[:self._c.agent_batch_size])) + \
                       tf.reduce_mean((q1_pred - tf.stop_gradient(q_target)) ** 2)
            qf2_loss = self.alpha * (tf.reduce_mean(q2_penalty) - tf.reduce_mean(q2_pred[:self._c.agent_batch_size])) + \
                       tf.reduce_mean((q2_pred - tf.stop_gradient(q_target)) ** 2)

            q_loss = qf1_loss + qf2_loss

        with tf.GradientTape() as actor_tape:
            new_obs_actions = self._actor(obs)
            q_new_actions = tf.reduce_min([self._qf1(tf.concat([obs, new_obs_actions], axis=-1)),
                                           self._qf2(tf.concat([obs, new_obs_actions], axis=-1))], axis=0)
            actor_loss = -tf.reduce_mean(q_new_actions)

        q_norm = self._qf_opt(q_tape, q_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        self._agent_step += 1

        if self._agent_step % self._c.log_every == 0:
            agent_summaries = dict()
            agent_summaries['agent/Q1_value'] = tf.reduce_mean(q1_pred)
            agent_summaries['agent/Q2_value'] = tf.reduce_mean(q2_pred)
            agent_summaries['agent/Q_target'] = tf.reduce_mean(q_target)
            agent_summaries['agent/Q_loss'] = q_loss
            agent_summaries['agent/actor_loss'] = actor_loss
            agent_summaries['agent/Q_grad_norm'] = q_norm
            agent_summaries['agent/actor_grad_norm'] = actor_norm
            self._write_summaries(agent_summaries, self._agent_step)

    def _update_target_critics(self):
        tau = tf.constant(self._c.tau)
        for source_weight, target_weight in zip(self._qf1.trainable_variables,
                                                self._target_qf1.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
        for source_weight, target_weight in zip(self._qf2.trainable_variables,
                                                self._target_qf2.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def _generate_latent_data(self, data):
        embed = self._encode(data)
        post, prior = self._dynamics.observe(embed, data['action'])
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}
        for key in post.keys():
            post[key] = tf.boolean_mask(post[key], data['mask'])
        start = post

        policy = lambda state: tf.stop_gradient(
            self._exploration(self._actor(self._dynamics.get_feat(state)), True))

        obs = [[] for _ in tf.nest.flatten(start)]
        next_obs = [[] for _ in tf.nest.flatten(start)]
        actions = []
        full_posts = [[[] for _ in tf.nest.flatten(start)] for _ in range(self._c.num_models)]
        prev = start

        for index in range(self._c.horizon):
            [o.append(l) for o, l in zip(obs, tf.nest.flatten(prev))]
            a = policy(prev)
            actions.append(a)
            for i in range(self._c.num_models):
                p = self._dynamics.img_step(prev, a, k=i)
                [o.append(l) for o, l in zip(full_posts[i], tf.nest.flatten(p))]
            prev = self._dynamics.img_step(prev, a, k=np.random.choice(self._c.num_models, 1)[0])
            [o.append(l) for o, l in zip(next_obs, tf.nest.flatten(prev))]

            obs = self._dynamics.get_feat(tf.nest.pack_sequence_as(start, [tf.stack(x, 0) for x in obs]))
            stoch = tf.nest.pack_sequence_as(start, [tf.stack(x, 0) for x in next_obs])['stoch']
            next_obs = self._dynamics.get_feat(tf.nest.pack_sequence_as(start, [tf.stack(x, 0) for x in next_obs]))
            actions = tf.stack(actions, 0)
            rewards = self._reward(next_obs).mode()
            if self._c.pcont:
                dones = 1.0 * (self._pcont(next_obs).mean().numpy() > self._c.done_treshold)
            else:
                dones = tf.zeros_like(rewards)

            dists = [self._dynamics.get_dist(
                tf.nest.pack_sequence_as(start, [tf.stack(x, 0) for x in full_posts[i]]))
                for i in range(self._c.num_models)]

            # Compute penalty based on specification
            if self._c.penalty_type == 'log_prob':
                log_prob_vars = tf.math.reduce_std(
                    tf.stack([d.log_prob(stoch) for d in dists], 0),
                    axis=0)
                modified_rewards = rewards - self.lmbd * log_prob_vars
            elif self._c.penalty_type == 'max_var':
                max_std = tf.reduce_max(
                    tf.stack([tf.norm(d.stddev(), 2, -1) for d in dists], 0),
                    axis=0)
                modified_rewards = rewards - self.lmbd * max_std
            elif self._c.penalty_type == 'mean':
                mean_prediction = tf.reduce_mean(tf.stack([d.mean() for d in dists], 0), axis=0)
                mean_disagreement = tf.reduce_mean(
                    tf.stack([tf.norm(d.mean() - mean_prediction, 2, -1) for d in dists], 0),
                    axis=0)
                modified_rewards = rewards - self.lmbd * mean_disagreement
            else:
                modified_rewards = rewards

            self.latent_buffer.add_samples(flatten(obs).numpy(),
                                           flatten(actions).numpy(),
                                           flatten(next_obs).numpy(),
                                           flatten(modified_rewards).numpy(),
                                           flatten(dones),
                                           sample_type='latent')

            obs = [[] for _ in tf.nest.flatten(start)]
            next_obs = [[] for _ in tf.nest.flatten(start)]
            actions = []
            full_posts = [[[] for _ in tf.nest.flatten(start)] for _ in range(self._c.num_models)]

            for key in prev.keys():
                prev[key] = tf.boolean_mask(prev[key], flatten(1.0 - dones))

    def _add_data(self, num_episodes=1):
        self._process_data_to_latent(num_episodes=num_episodes)
        self._generate_latent_data(next(self._dataset))

    def _process_data_to_latent(self, num_episodes=None):
        if num_episodes is None:
            num_episodes, _ = tools.count_episodes(self._c.datadir)

        for _ in range(num_episodes):
            filename = next(self._episode_iterator)
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f'Could not load episode: {e}')
                continue

            obs = preprocess_raw(episode, self._c)
            if not self._c.pcont:
                obs['terminal'] = tf.zeros_like(obs['reward'])
            with tf.GradientTape(watch_accessed_variables=False) as _:
                embed = self._encode(obs)
                post, prior = self._dynamics.observe(tf.expand_dims(embed, 0),
                                                     tf.expand_dims(obs['action'], 0))
                feat = flatten(self._dynamics.get_feat(post))
                self.latent_buffer.add_samples(feat.numpy()[:-1],
                                               obs['action'].numpy()[1:],
                                               feat.numpy()[1:],
                                               obs['reward'].numpy()[1:],
                                               obs['terminal'].numpy()[1:],
                                               sample_type='real')

    def _image_summaries(self, data, embed, image_pred, step=None, prefix='train'):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:6, 5:], init)
        openl = self._decode(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error_prior = (model - truth + 1) / 2
        error_posterior = (recon + 0.5 - truth + 1) / 2
        openl = tf.concat([truth, recon + 0.5, model, error_prior, error_posterior], 2)
        with self._writer.as_default():
            tools.video_summary('agent/' + prefix, openl.numpy(), step=step)

    def _write_summaries(self, metrics, step=None):
        step = int(step)
        metrics = [(k, float(v)) for k, v in metrics.items()]
        with self._writer.as_default():
            tf.summary.experimental.set_step(step)
            [tf.summary.scalar(k, m, step=step) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


def preprocess_raw(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()

    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        if 'image_128' in obs.keys():
            obs['image_128'] = tf.cast(obs['image_128'], dtype) / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
        for k in obs.keys():
            obs[k] = tf.cast(obs[k], dtype)
    return obs


def flatten(x):
    return tf.reshape(x, [-1] + list(x.shape[2:]))


def preprocess_latent(batch):
    dtype = prec.global_policy().compute_dtype
    batch = batch.copy()
    with tf.device('cpu:0'):
        for key in batch.keys():
            batch[key] = tf.cast(batch[key], dtype)
    return batch


def count_steps(datadir, config):
    return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1000, load_episodes=1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: tools.load_episodes(directory, config.train_steps,
                                            config.model_batch_length, config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.model_batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess_raw, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def episode_iterator(datadir, config):
    while True:
        filenames = list(datadir.glob('*.npz'))
        for filename in list(filenames):
            yield filename


def summarize_episode(episode, config, datadir, writer, prefix):
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1)]
    with writer.as_default():  # Env might run in a different thread.
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


def make_env(config, writer, prefix, datadir, store):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task)
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
    elif suite == 'gym':
        env = wrappers.Gym(task, config, size=(128, 128))
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
    elif task == 'door':
        env = wrappers.DoorOpen(config, size=(128, 128))
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
    elif task == 'drawer':
        env = wrappers.DrawerOpen(config, size=(128, 128))
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    if prefix == 'test':
        callbacks.append(
            lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    return env


def main(config):
    print(config)

    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    config.logdir = config.logdir / config.task
    config.logdir = config.logdir / 'seed_{}'.format(config.seed)
    config.logdir.mkdir(parents=True, exist_ok=True)
    tf_dir = config.logdir / 'tensorboard'
    writer = tf.summary.create_file_writer(str(tf_dir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()

    # Create environments.
    train_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'train', '.', store=False), config.parallel)
                  for _ in range(config.envs)]
    test_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'test', '.', store=False), config.parallel)
                 for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Train and regularly evaluate the agent.
    agent = Lompo(config, config.datadir, actspace, writer)

    if agent._c.load_model:
        agent.load_model(config.loaddir / 'final_model')
        print('Load pretrained model')
    else:
        agent.fit_model(agent._c.model_train_steps)
        agent.save_model(config.logdir)

    if agent._c.load_buffer:
        agent.latent_buffer.load(agent._c.loaddir / 'buffer.h5py')
    else:
        agent._process_data_to_latent()
        agent.latent_buffer.save(agent._c.logdir / 'buffer.h5py')

    if agent._c.load_agent:
        agent.load_agent(config.loaddir)
        print('Load pretrained actor')

    while agent.latent_buffer._latent_stored_steps < agent._c.start_training:
        agent._generate_latent_data(next(agent._dataset))

    while agent._agent_step < int(config.agent_train_steps):
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        agent.train_agent(agent._c.agent_iters_per_step)

        if config.sample:
            agent._add_data(num_episodes=1)
        else:
            agent._process_data_to_latent(num_episodes=1)

    for env in train_envs + test_envs:
        env.close()


if __name__ == '__main__':
    try:
        import colored_traceback

        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    main(parser.parse_args())
