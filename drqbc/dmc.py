# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

from envs.distracting_control.suite import distracting_wrapper
import envs.fb_mtenv_dmc as fb_mtenv_dmc


def get_unique_int(difficulty: str) -> int:
    return int.from_bytes(f'{difficulty}_0'.encode(), 'little') % (2 ** 31)


distracting_kwargs_lookup = {
    'easy': {'difficulty': 'easy', 'fixed_distraction': False},
    'medium': {'difficulty': 'medium', 'fixed_distraction': False},
    'hard': {'difficulty': 'hard', 'fixed_distraction': False},
    'fixed_easy': {'difficulty': 'easy', 'fixed_distraction': True, 'color_seed': get_unique_int('easy'),
                   'background_seed': get_unique_int('easy'), 'camera_seed': get_unique_int('easy')},
    'fixed_medium': {'difficulty': 'medium', 'fixed_distraction': True, 'color_seed': get_unique_int('medium'),
                     'background_seed': get_unique_int('medium'), 'camera_seed': get_unique_int('medium')},
    'fixed_hard': {'difficulty': 'hard', 'fixed_distraction': True, 'color_seed': get_unique_int('hard'),
                   'background_seed': get_unique_int('hard'), 'camera_seed': get_unique_int('hard')},
}

multitask_modes = [f'len_{i}' for i in range(1, 11, 1)]


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, distracting_mode: str = None, multitask_mode: str = None):
    pixel_hw = 84
    if 'offline' in name:
        name = '_'.join(name.split('_')[1:3])
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    # make sure reward is not visualized
    if multitask_mode is None:
        if (domain, task) in suite.ALL_TASKS:
            env = suite.load(domain,
                             task,
                             task_kwargs={'random': seed},
                             visualize_reward=False)
            pixels_key = 'pixels'
        else:
            name = f'{domain}_{task}_vision'
            env = manipulation.load(name, seed=seed)
            pixels_key = 'front_close'
    else:
        assert multitask_mode in multitask_modes, 'Unrecognised length setting'
        idx = multitask_mode.split('_', 1)[1]

        if domain == 'walker' and task == 'walk':
            xml = f'len_{idx}'
        elif domain == 'cheetah' and task == 'run':
            xml = f'torso_length_{idx}'
        else:
            raise Exception

        env = fb_mtenv_dmc.load(
            domain_name=domain,
            task_name=task,
            task_kwargs={'xml_file_id': xml, 'random': seed},
            visualize_reward=False,
        )
        pixels_key = 'pixels'

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=pixel_hw, width=pixel_hw, camera_id=camera_id)
        if distracting_mode is not None:
            assert distracting_mode in distracting_kwargs_lookup, 'Unrecognised distraction'
            kwargs = distracting_kwargs_lookup[distracting_mode]
            kwargs['pixels_only'] = True
            kwargs['render_kwargs'] = render_kwargs
            kwargs['background_dataset_path'] = "DAVIS/JPEGImages/480p/"
            env = distracting_wrapper(
                env,
                domain,
                **kwargs
            )
        else:
            env = pixels.Wrapper(env,
                                 pixels_only=True,
                                 render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
