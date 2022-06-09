import h5py
import numpy as np


class Buffer():
    def __init__(self):
        self._obs = None
        self._actions = None
        self._rewards = None
        self._next_obs = None
        self._terminals = None


class LatentReplayBuffer(object):
    def __init__(self,
                 real_size: int,
                 latent_size: int,
                 obs_dim: int,
                 action_dim: int,
                 immutable: bool = False,
                 load_from: str = None,
                 silent: bool = False,
                 seed: int = 0):

        self.immutable = immutable

        self.buffers = dict()
        self.sizes = {'real': real_size, 'latent': latent_size}
        for key in ['real', 'latent']:
            self.buffers[key] = Buffer()
            self.buffers[key]._obs = np.full((self.sizes[key], obs_dim), float('nan'), dtype=np.float32)
            self.buffers[key]._actions = np.full((self.sizes[key], action_dim), float('nan'), dtype=np.float32)
            self.buffers[key]._rewards = np.full((self.sizes[key], 1), float('nan'), dtype=np.float32)
            self.buffers[key]._next_obs = np.full((self.sizes[key], obs_dim), float('nan'), dtype=np.float32)
            self.buffers[key]._terminals = np.full((self.sizes[key], 1), float('nan'), dtype=np.float32)

        self._real_stored_steps = 0
        self._real_write_location = 0

        self._latent_stored_steps = 0
        self._latent_write_location = 0

        self._stored_steps = 0
        self._random = np.random.RandomState(seed)

    @property
    def obs_dim(self):
        return self._obs.shape[-1]

    @property
    def action_dim(self):
        return self._actions.shape[-1]

    def __len__(self):
        return self._stored_steps

    def save(self, location: str):
        f = h5py.File(location, 'w')
        f.create_dataset('obs', data=self.buffers['real']._obs[:self._real_stored_steps], compression='lzf')
        f.create_dataset('actions', data=self.buffers['real']._actions[:self._real_stored_steps], compression='lzf')
        f.create_dataset('rewards', data=self.buffers['real']._rewards[:self._real_stored_steps], compression='lzf')
        f.create_dataset('next_obs', data=self.buffers['real']._next_obs[:self._real_stored_steps], compression='lzf')
        f.create_dataset('terminals', data=self.buffers['real']._terminals[:self._real_stored_steps], compression='lzf')
        f.close()

    def load(self, location: str):
        with h5py.File(location, "r") as f:
            obs = np.array(f['obs'])
            self._real_stored_steps = obs.shape[0]
            self._real_write_location = obs.shape[0] % self.sizes['real']

            self.buffers['real']._obs[:self._real_stored_steps] = np.array(f['obs'])
            self.buffers['real']._actions[:self._real_stored_steps] = np.array(f['actions'])
            self.buffers['real']._rewards[:self._real_stored_steps] = np.array(f['rewards'])
            self.buffers['real']._next_obs[:self._real_stored_steps] = np.array(f['next_obs'])
            self.buffers['real']._terminals[:self._real_stored_steps] = np.array(f['terminals'])

    def add_samples(self, obs_feats, actions, next_obs_feats, rewards, terminals, sample_type='latent'):
        if sample_type == 'real':
            for obsi, actsi, nobsi, rewi, termi in zip(obs_feats, actions, next_obs_feats, rewards, terminals):
                self.buffers['real']._obs[self._real_write_location] = obsi
                self.buffers['real']._actions[self._real_write_location] = actsi
                self.buffers['real']._next_obs[self._real_write_location] = nobsi
                self.buffers['real']._rewards[self._real_write_location] = rewi
                self.buffers['real']._terminals[self._real_write_location] = termi

                self._real_write_location = (self._real_write_location + 1) % self.sizes['real']
                self._real_stored_steps = min(self._real_stored_steps + 1, self.sizes['real'])

        else:
            for obsi, actsi, nobsi, rewi, termi in zip(obs_feats, actions, next_obs_feats, rewards, terminals):
                self.buffers['latent']._obs[self._latent_write_location] = obsi
                self.buffers['latent']._actions[self._latent_write_location] = actsi
                self.buffers['latent']._next_obs[self._latent_write_location] = nobsi
                self.buffers['latent']._rewards[self._latent_write_location] = rewi
                self.buffers['latent']._terminals[self._latent_write_location] = termi

                self._latent_write_location = (self._latent_write_location + 1) % self.sizes['latent']
                self._latent_stored_steps = min(self._latent_stored_steps + 1, self.sizes['latent'])

        self._stored_steps = self._real_stored_steps + self._latent_stored_steps

    def sample(self, batch_size, return_dict: bool = False):
        real_idxs = self._random.choice(self._real_stored_steps, batch_size)
        latent_idxs = self._random.choice(self._latent_stored_steps, batch_size)

        obs = np.concatenate([self.buffers['real']._obs[real_idxs],
                              self.buffers['latent']._obs[latent_idxs]], axis=0)
        actions = np.concatenate([self.buffers['real']._actions[real_idxs],
                                  self.buffers['latent']._actions[latent_idxs]], axis=0)
        next_obs = np.concatenate([self.buffers['real']._next_obs[real_idxs],
                                   self.buffers['latent']._next_obs[latent_idxs]], axis=0)
        rewards = np.concatenate([self.buffers['real']._rewards[real_idxs],
                                  self.buffers['latent']._rewards[latent_idxs]], axis=0)
        terminals = np.concatenate([self.buffers['real']._terminals[real_idxs],
                                    self.buffers['latent']._terminals[latent_idxs]], axis=0)

        data = {
            'obs': obs,
            'actions': actions,
            'next_obs': next_obs,
            'rewards': rewards,
            'terminals': terminals
        }

        return data
