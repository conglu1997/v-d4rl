import copy
import datetime
import io
import pathlib
import pickle
import uuid

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow_probability import distributions as tfd


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Module(tf.Module):
    def save(self, filename):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        with pathlib.Path(filename).open('wb') as f:
            pickle.dump(values, f)

    def load(self, filename):
        with pathlib.Path(filename).open('rb') as f:
            values = pickle.load(f)
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


def video_summary(name, video, step=None, fps=20):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def simulate(agent, envs, steps=0, episodes=0, state=None):
    # Initialize or unpack simulation state.
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), np.bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # Reset envs if necessary.
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            promises = [envs[i].reset(blocking=False) for i in indices]
            for index, promise in zip(indices, promises):
                obs[index] = promise()
        # Step agents.
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state)
        action = np.array(action)
        assert len(action) == len(envs)
        # Step envs.
        promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
        obs, _, done = zip(*[p()[:3] for p in promises])
        obs = list(obs)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += (done * length).sum()
        length *= (1 - done)
    # Return new state to allow resuming the simulation.
    return step - steps, episode - episodes, done, length, obs, agent_state


def count_episodes(directory):
    filenames = directory.glob('*.npz')
    lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode['reward'])
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())


def load_episodes(directory, rescan, length=None, balance=False, seed=0, load_episodes=1000):
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    filenames = list(directory.glob('*.npz'))
    load_episodes = min(len(filenames), load_episodes)
    if load_episodes is None:
        load_episodes = int(count_episodes(directory)[0] / 20)

    while True:
        cache = {}
        for filename in random.choice(list(directory.glob('*.npz')),
                                      load_episodes,
                                      replace=False):
            try:
                with filename.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys() if k not in ['image_128']}
                    # episode['reward'] = copy.deepcopy(episode['success'])
                    if 'discount' not in episode:
                        episode['discount'] = np.where(episode['is_terminal'], 0., 1.)
            except Exception as e:
                print(f'Could not load episode: {e}')
                continue
            cache[filename] = episode

        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = copy.deepcopy(cache[keys[index]])
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 0:
                    for key in episode.keys():
                        shape = episode[key].shape
                        episode[key] = np.concatenate([episode[key],
                                                       np.zeros([abs(available)] + list(shape[1:]))],
                                                      axis=0)
                    episode['mask'] = np.ones(length)
                    episode['mask'][available:] = 0.0
                elif available > 0:
                    if balance:
                        index = min(random.randint(0, total), available)
                    else:
                        index = int(random.randint(0, available))
                    episode = {k: v[index: index + length] for k, v in episode.items()}
                    episode['mask'] = np.ones(length)
                else:
                    episode['mask'] = np.ones_like(episode['reward'])
            else:
                episode['mask'] = np.ones_like(episode['reward'])
            yield episode


class Adam(tf.Module):
    def __init__(self, name, modules, lr, clip=None, wd=None, wdpattern=r'.*'):
        self._name = name
        self._modules = modules
        self._clip = clip
        self._wd = wd
        self._wdpattern = wdpattern
        self._opt = tf.optimizers.Adam(lr)

    @property
    def variables(self):
        return self._opt.variables()

    def __call__(self, tape, loss):
        variables = [module.variables for module in self._modules]
        self._variables = tf.nest.flatten(variables)
        assert len(loss.shape) == 0, loss.shape
        grads = tape.gradient(loss, self._variables)
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        self._opt.apply_gradients(zip(grads, self._variables))
        return norm


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


def _mnd_sample(self, sample_shape=(), seed=None, name='sample'):
    return tf.random.normal(
        tuple(sample_shape) + tuple(self.event_shape),
        self.mean(), self.stddev(), self.dtype, seed, name)


tfd.MultivariateNormalDiag.sample = _mnd_sample
