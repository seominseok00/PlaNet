import cv2
import numpy as np
import torch
import safety_gymnasium


def preprocess_observation(obs, bit_depth):
    """[0, 255] -> [-0.5, 0.5] + Noise (In-place)"""
    # 1. Quantization and scaling to [-0.5, 0.5]
    obs.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # 2. Add dequantization noise
    obs.add_(torch.rand_like(obs).div_(2 ** bit_depth))

def postprocess_observation(obs, bit_depth):
    """[-0.5, 0.5] -> [0, 255] uint8"""
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    # obs.transpose(2, 0, 1)
    img = np.clip(np.floor((obs + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1)
    return img.astype(np.uint8)

def _images_to_observation(img, bit_depth):
    # 1. transpose (H, W, C) -> (C, H, W)
    img_transposed = img.transpose(2, 0, 1).copy()
    # 2. to tensor and preprocess
    img_tensor = torch.tensor(img_transposed, dtype=torch.float32)
    preprocess_observation(img_tensor, bit_depth)
    return img_tensor.unsqueeze(dim=0)  # (C, H, W) -> (1, C, H, W)


class SafetyGymEnv():
    def __init__(self, env_id, size=(64, 64), action_repeat=5, bit_depth=5):
        self._action_repeat = action_repeat
        self._bit_depth = bit_depth
        self._camera_name = 'vision'
        self._size = size
        self._env = safety_gymnasium.make(env_id, render_mode='rgb_array', width=size[0], height=size[1])

    def reset(self):
        self.t = 0
        # Note: env observation is ignored, we use only rendered image
        _obs, info = self._env.reset()
        image = self._env.task.render(width=self._size[0], height=self._size[1],
                                      mode='rgb_array', camera_name=self._camera_name, cost={})
        return _images_to_observation(image, self._bit_depth)

    def step(self, action):
        total_reward = 0
        total_cost = 0
        
        # Repeat action
        for _ in range(self._action_repeat):
            # Note: env observation is ignored, we use only rendered image
            _obs, reward, cost, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            total_cost += cost
            done = terminated or truncated
            self.t += 1
            if done:
                break
        
        # Render image after action repeats
        image = self._env.task.render(width=self._size[0], height=self._size[1], 
                                      mode='rgb_array', camera_name=self._camera_name, cost={})
        obs = _images_to_observation(image, self._bit_depth)

        return obs, total_reward, total_cost, done
    
    def render(self):
        image = self._env.task.render(width=self._size[0], height=self._size[1], 
                                      mode='rgb_array', camera_name=self._camera_name, cost={})
        cv2.imshow('screen', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    @property
    def observation_space(self):
        return (3, self._size[0], self._size[1])

    @property
    def action_space(self):
        return self._env.action_space.shape[0]
    
    @property
    def action_range(self):
        return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])
    
    def sample_random_action(self):
        return self._env.action_space.sample()

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()


class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)
    
    def step(self, actions):
        # Cast bool to int tensor -> nonzero() returns 2D indices -> (num_nonzeros, ndim) -> Slice [:, 0] to get a 1D index mask (num_nonzeros, )
        '''
        torch.nonzero() example
        - 1D: [T, F, T] -> [[0], [2]] --[:,0]--> [0, 2]
        - 2D: [[T, F], [F, T]] -> [[0, 0], [1, 1]]
        '''
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, costs, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
        self.dones = dones
        
        observations, rewards, costs, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(costs, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
        
        observations[done_mask] = 0
        rewards[done_mask] = 0
        costs[done_mask] = 0
        
        return observations, rewards, costs, dones
    
    def close(self):
        [env.close() for env in self.envs]