import atexit
import functools
import sys
import traceback

import gym
import numpy as np


class DrawerOpen:
    def __init__(self, config, size=(128, 128)):
        import metaworld
        import mujoco_py
        self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2.SawyerDrawerOpenEnvV2()
        self._env._last_rand_vec = np.array([-0.1, 0.9, 0.0])
        self._env._set_task_called = True
        self.size = size

        # Setup camera in environment
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer.cam.elevation = -22.5
        self.viewer.cam.azimuth = 15
        self.viewer.cam.distance = 0.75
        self.viewer.cam.lookat[0] = -0.15
        self.viewer.cam.lookat[1] = 0.7
        self.viewer.cam.lookat[2] = 0.10

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        obs = {'state': state, 'image': img}
        reward = 1.0 * info['success']
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        state = self._env.reset()
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        if self.use_transform:
            img = img[self.pad:-self.pad, self.pad:-self.pad, :]
        obs = {'state': state, 'image': img}
        return obs

    def render(self, mode, width=128, height=128):
        self.viewer.render(width=width, height=width)
        img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
        img = img[::-1]
        return img


class Hammer:
    def __init__(self, config, size=(128, 128)):
        import metaworld
        import mujoco_py
        self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_hammer_v2.SawyerHammerEnvV2()
        self._env._last_rand_vec = np.array([-0.06, 0.4, 0.02])
        self._env._set_task_called = True
        self.size = size

        # Setup camera in environment
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 137.5
        self.viewer.cam.distance = 0.9
        self.viewer.cam.lookat[0] = -0.
        self.viewer.cam.lookat[1] = 0.6
        self.viewer.cam.lookat[2] = 0.175

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        obs = {'state': state, 'image': img}
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        obs = {'state': state, 'image': img}
        return obs

    def render(self, mode, width=128, height=128):
        self.viewer.render(width=width, height=width)
        img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
        img = img[::-1]
        return img


class DoorOpen:
    def __init__(self, config, size=(128, 128)):
        import metaworld
        import mujoco_py
        self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_v2.SawyerDoorEnvV2()
        self._env._last_rand_vec = np.array([0.0, 1.0, .1525])
        self._env._set_task_called = True
        self.size = size

        # Setup camera in environment
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer.cam.elevation = -12.5
        self.viewer.cam.azimuth = 115
        self.viewer.cam.distance = 1.05
        self.viewer.cam.lookat[0] = 0.075
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.15

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        obs = {'state': state, 'image': img}
        reward = 1.0 * info['success']
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        img = self.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        obs = {'state': state, 'image': img}
        return obs

    def render(self, mode, width=128, height=128):
        self.viewer.render(width=width, height=width)
        img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
        img = img[::-1]
        return img


class Gym:
    def __init__(self, name, config, size=(64, 64)):
        self._env = gym.make(name)
        self.size = size
        self.use_transform = config.use_transform
        self.pad = int(config.pad / 2)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        img = self._env.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        if self.use_transform:
            img = img[self.pad:-self.pad, self.pad:-self.pad, :]
        obs = {'state': state, 'image': img}
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        img = self._env.render(mode='rgb_array', width=self.size[0], height=self.size[1])
        if self.use_transform:
            img = img[self.pad:-self.pad, self.pad:-self.pad, :]
        obs = {'state': state, 'image': img}
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(mode='rgb_array', width=self.size[0], height=self.size[1])


class DeepMindControl:
    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class Collect:
    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))
        self._episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self._env.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(np.isfinite(env.action_space.low),
                                    np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:
    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class Async:
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, ctor, strategy='process'):
        self._strategy = strategy
        if strategy == 'none':
            self._env = ctor()
        elif strategy == 'thread':
            import multiprocessing.dummy as mp
        elif strategy == 'process':
            import multiprocessing as mp
        else:
            raise NotImplementedError(strategy)
        if strategy != 'none':
            self._conn, conn = mp.Pipe()
            self._process = mp.Process(target=self._worker, args=(ctor, conn))
            atexit.register(self.close)
            self._process.start()
        self._obs_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._obs_space:
            self._obs_space = self.__getattr__('observation_space')
        return self._obs_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        if self._strategy == 'none':
            return getattr(self._env, name)
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        blocking = kwargs.pop('blocking', True)
        if self._strategy == 'none':
            return functools.partial(getattr(self._env, name), *args, **kwargs)
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        promise = self._receive
        return promise() if blocking else promise

    def close(self):
        if self._strategy == 'none':
            try:
                self._env.close()
            except AttributeError:
                pass
            return
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        return self.call('step', action, blocking=blocking)

    def reset(self, blocking=True):
        return self.call('reset', blocking=blocking)

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except ConnectionResetError:
            raise RuntimeError('Environment worker crashed.')
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError(f'Received message of unexpected type {message}')

    def _worker(self, ctor, conn):
        try:
            env = ctor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError(f'Received message of unknown type {message}')
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print(f'Error in environment process: {stacktrace}')
            conn.send((self._EXCEPTION, stacktrace))
        conn.close()
