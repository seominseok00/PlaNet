import os
os.environ['MUJOCO_GL'] = 'glfw'

import time
from math import inf
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

from env import SafetyGymEnv, EnvBatcher
from memory import ExperienceReplay
from models import TransitionModel, ObservationModel, RewardModel, Encoder, bottle
from planner import MPCPlanner

seed = 1
env_id = "SafetyPointGoal1-v0"
max_episode_length = 1000
experience_size = 1000000
activation = F.relu
embedding_size = 1024
hidden_size = 200
belief_size = 200
state_size = 30
action_repeat = 4
action_noise = 0.3
episodes = 1000
seed_episodes = 5
collect_interval = 100  # collect_interval = 2
batch_size = 50
chunk_size = 50
overshooting_distance = 50
overshooting_kl_beta = 0
overshooting_reward_scale = 0
global_kl_beta = 0
free_nats = 3
bit_depth = 5
learning_rate = 1e-3
grad_clip_norm = 1000
planning_horizon = 12
optimization_iters = 10
candidates = 1000
top_candidates = 100
test_interval = 25      # test_interval = 1
test_episodes = 10      # test_episodes = 2
checkpoint_interval = 50


#=====================================================================#
#                                Setup                                #
#=====================================================================#
id = datetime.now().strftime('%Y-%m-%d-%H-%M-') + env_id
result_dir = os.path.join('results', id)
os.makedirs(result_dir, exist_ok=True)

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
else:
    device = torch.device('cpu')

print("🚀 Training on device: ", device)



metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'train_costs': [], 'test_episodes': [], 'test_rewards': [], 'test_costs': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}


#=====================================================================#
#     Initialize training environment and experience replay memory    #
#=====================================================================#
env = SafetyGymEnv(env_id=env_id, size=(64, 64), action_repeat=action_repeat, bit_depth=bit_depth)
buf = ExperienceReplay(size=experience_size, obs_shape=env.observation_space, act_shape=env.action_space, bit_depth=bit_depth, device=device)

for seed_episode in range(1, seed_episodes + 1):
    obs, done, t = env.reset(), False, 0
    while not done:
        action = env.sample_random_action()
        next_obs, reward, cost, done = env.step(action)
        buf.append(obs, action, reward, done)
        obs = next_obs
        t += 1
    metrics['steps'].append(t * action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(seed_episode)

#=====================================================================#
#                           Initialize Model                          #
#=====================================================================#
transition_model = TransitionModel(
    belief_size=belief_size, 
    state_size=state_size, 
    action_size=env.action_space, 
    hidden_size=hidden_size, 
    embedding_size=embedding_size, 
    activation=activation
).to(device)

observation_model = ObservationModel(
    belief_size=belief_size, 
    state_size=state_size, 
    embedding_size=embedding_size
).to(device)

reward_model = RewardModel(
    belief_size=belief_size, 
    state_size=state_size, 
    hidden_size=hidden_size
).to(device)

encoder = Encoder(embedding_size=embedding_size).to(device)

param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(param_list, lr=learning_rate, eps=1e-4)

planner = MPCPlanner(
    action_size=env.action_space, 
    planning_horizon=planning_horizon, 
    optimization_iters=optimization_iters, 
    candidates=candidates, 
    top_candidates=top_candidates, 
    transition_model=transition_model, 
    reward_model=reward_model, 
    min_action=env.action_range[0], 
    max_action=env.action_range[1]
).to(device)

kl_free_nats = torch.full((1, ), free_nats, dtype=torch.float32, device=device)

transition_model.eval()
reward_model.eval()
encoder.eval()


def update_belief_and_act(env, planner, transition_model, encoder, belief, posterior_state, action, observation, action_noise, min_action=-inf, max_action=inf, explore=False):    
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = action + action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    action.clamp_(min=min_action, max=max_action)  # Clip action range
    next_observation, reward, cost, done = env.step(action.detach().cpu().numpy() if isinstance(env, EnvBatcher) else action[0].detach().cpu().numpy())
    return belief, posterior_state, action, next_observation, reward, cost, done


start_time = time.time()

#=====================================================================#
#                               Training                              #
#=====================================================================#
outer_pbar = tqdm(range(metrics['episodes'][-1] + 1, episodes + 1), total=episodes, initial=metrics['episodes'][-1] + 1, desc="[Overall Progress]", position=0)
for episode in outer_pbar:
    losses = []

    train_pbar = tqdm(range(collect_interval), desc=f'  ┗ [Train] Episode {episode} Update', position=1, leave=False)
    for s in train_pbar:
        
        # Draw sequence chunks {(o_t, a_t, r_{t + 1}, d_{t + 1})} ~ D uniformly at random from the dataset
        observations, actions, rewards, nonterminals = buf.sample(batch_size, chunk_size)

        '''
        Q. Why start with zeros if the chunk is from the middle of episode?
        -> We don't store hidden states in the buffer becuase they become stale as the model is updated. The model recovers the true latent state by processing the few steps.
        '''
        init_belief, init_state = torch.zeros(batch_size, belief_size, device=device), torch.zeros(batch_size, state_size, device=device)


        # Update belief/state using posterior from previous belief/state, previous action and current observation
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(prev_state=init_state, actions=actions[:-1], prev_belief=init_belief, observations=bottle(encoder, (observations[1:], )), nonterminals=nonterminals[:-1])

    
        # Calculate observation likelihood, reward likelihood and KL losses; sum over final dims(channel, width, height), average over batch and time (original implementation, though paper seems to miss 1/T scailing?)
        observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        
        reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))
        
        kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), kl_free_nats).mean(dim=(0, 1))  # Note that normalization by overshooting distance and weighting by overshooting distance cancel out

        # Update model parameters
        optimizer.zero_grad()
        (observation_loss + reward_loss + kl_loss).backward()
        nn.utils.clip_grad_norm_(param_list, grad_clip_norm, norm_type=2)
        optimizer.step()

        train_pbar.set_postfix({
            'obs_loss': f'{observation_loss.item():.4f}',
            'rew_loss': f'{reward_loss.item():.4f}',
            'kl_loss': f'{kl_loss.item():.4f}'
        })

        # Store losses (0) observation loss (1) reward loss (2) KL loss
        losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])

    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    

    #=====================================================================#
    #                           Data Collection                           #
    #=====================================================================#
    with torch.no_grad():
        observation, total_reward, total_cost = env.reset(), 0, 0
        belief, posterior_state, action = torch.zeros(1, belief_size, device=device), torch.zeros(1, state_size, device=device), torch.zeros(1, env.action_space, device=device)

        collect_pbar = tqdm(range(max_episode_length // action_repeat), desc=f'  ┗ [Collect] Episode {episode}', position=1, leave=False)
        for t in collect_pbar:
            belief, posterior_state, action, next_observation, reward, cost, done = update_belief_and_act(env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=device), action_noise, env.action_range[0], env.action_range[1])

            buf.append(observation, action[0].detach().cpu().numpy(), reward, done)
            
            total_reward += reward
            total_cost += cost

            observation = next_observation
            collect_pbar.set_postfix({'reward': f'{total_reward:.1f}', 'cost': f'{total_cost:.1f}'})

            if done:
                collect_pbar.close()
                break

    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    metrics['train_costs'].append(total_cost)

    #====================================================================#
    #                             Test Model                             #
    #====================================================================#
    if episode % test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()

        # Initialize parallelised test environments
        test_envs = EnvBatcher(env_class=SafetyGymEnv, env_args=(env_id, (64, 64), action_repeat, bit_depth), env_kwargs={}, n=test_episodes)

        with torch.no_grad():
            observation, total_rewards, total_costs = test_envs.reset(), np.zeros((test_episodes, )), np.zeros((test_episodes, ))

            belief, posterior_state, action = torch.zeros(test_episodes, belief_size, device=device), torch.zeros(test_episodes, state_size, device=device), torch.zeros(test_episodes, env.action_space, device=device)

            test_pbar = tqdm(range(max_episode_length // action_repeat), desc=f'  ┗ [Test] Episode {episode}', position=1, leave=False)

            for t in test_pbar:
                belief, posterior_state, action, next_observation, reward, cost, done = update_belief_and_act(test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=device), action_noise, env.action_range[0], env.action_range[1])

                total_rewards += reward.detach().cpu().numpy()
                total_costs += cost.detach().cpu().numpy()

                observation = next_observation

                if done.sum().item() == test_episodes:
                    test_pbar.close()
                    break

        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        metrics['test_costs'].append(total_costs.tolist())

        outer_pbar.set_postfix({
            'test_reward': f'{np.mean(total_rewards):.1f}',
            'test_cost': f'{np.mean(total_costs):.1f}'
        })

        torch.save(metrics, os.path.join(result_dir, 'metrics.pth'))

        
        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()

        # Close test environments
        test_envs.close()

    
    # Checkpoint models
    if episode % checkpoint_interval == 0:
        torch.save({'transition_model': transition_model.state_dict(), 'observation_model': observation_model.state_dict(), 'reward_model': reward_model.state_dict(), 'encoder': encoder.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(result_dir, 'models_%d.pth' % episode))
    

# Close training environment
env.close()

end_time = time.time()
print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))