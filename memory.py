import numpy as np
import torch

from env import preprocess_observation, postprocess_observation

class ExperienceReplay():
    def __init__(self, size, obs_shape, act_shape, bit_depth, device):
        self.bit_depth = bit_depth
        self.device = device
        self.full = False
        self.idx = 0
        self.steps, self.episodes = 0, 0
        self.size = size

        self.observations = np.empty((size, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((size, act_shape), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)

    def append(self, obs, act, reward, done):
        self.observations[self.idx] = postprocess_observation(obs, self.bit_depth)
        self.actions[self.idx] = act
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        if done:
            self.episodes += 1
        
    def _sample_indices(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]
        return idxs
    
    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # (n * L, )
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))  # observations: (n * L, C, H, W)
        preprocess_observation(observations, self.bit_depth)
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)
    
    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray([self._sample_indices(L) for _ in range(n)]), n, L)  # [self._sample_indices(L) for _ in range(n)]: (n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]