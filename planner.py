from math import inf

import torch
from torch import jit

class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimization_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimization_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
        super(MPCPlanner, self).__init__()
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        
        self.planning_horizon = planning_horizon
        self.optimization_iters = optimization_iters
        
        self.candidates = candidates
        self.top_candidates = top_candidates
        
        self.transition_model = transition_model
        self.reward_model = reward_model

    @jit.script_method
    def forward(self, belief:torch.Tensor, state:torch.Tensor) -> torch.Tensor:
        '''
        belief: (batch_size, belief_size)
        state: (batch_size, state_size)
        '''
        
        B, H, Z = belief.size(0), belief.size(1), state.size(1)

        '''
        - unsqueeze: (B, H) -> (B, 1, H)
        - expand:    (B, 1, H) -> (B, C, H)
        - reshape:   (B, C, H) -> (B * C, H)
        '''
        belief = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H)
        state = state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)

        # Initialize action distribution
        '''
        Q. Why planning_horizon is added as the first dimension?
        -> Because we need to sample actions for each time step in the planning horizon, So we need to have a separate distribution for each time step. (because each time step's optimal action may differ)

        Q. Why is the dimension '1' added after the batch dimension?
        -> Because CEM maintains a single distribution (mean and std) per batch and time step. This '1' acts as a placeholder for broadcasting, allowing us to efficiently sample multiple candidates from that single distribution.
        '''
        action_mean = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device)  # (P, B, 1, A)
        action_std_dev = torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)  # (P, B, 1, A)

        for _ in range(self.optimization_iters):
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device))  # (P, B, 1, A) -> (P, B, C, A)
            actions = actions.view(self.planning_horizon, B * self.candidates, self.action_size)  # (P, B * C, A)
            actions = torch.clamp(actions, self.min_action, self.max_action)
    
            # Sample next states
            '''
            Transition model 
            
            input
            - state: (B * C, Z)
            - belief: (B * C, H)
            - actions: (P, B * C, A)

            output
            - beliefs: (P, B * C, H)
            - states: (P, B * C, Z)
            '''
            beliefs, states, _, _ = self.transition_model(state, actions, belief)

            # Calculate expected returns (technically sum of rewards over planning horizon)
            '''
            Reward model 
            
            input
            - beliefs: (P * B * C, H)
            - states: (P * B * C, Z)

            output
            - returns: (P, B * C)

            '''
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1)            
            returns = returns.sum(dim=0)  # (P, B * C) -> (B * C)

            
            # Re-fit belief to the K best action sequences
            '''
            - reshape: (B * C) -> (B, C)
            - topk: (B, C) -> (B, K)
            '''
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim = 1, largest=True, sorted=False)
            
            '''
            - torch.arange(0, B): [0, 1, 2, ..., B - 1]  -> (B, )
            - torch.arange(0, B).unsqueeze(dim=1): (B, ) -> (B, 1)
            - topk += self.candidates * torch.arange(0, B).unsqueeze(dim=1): (B, K)
            
            Q. Why add (B, 1) offsets to (B, K) topk?
            -> To convert 'relative indicies' (0 to C - 1) into 'absolute indicies' (0 to B * C - 1).

            Example (B=2, C=1000, K=3)

            1. topk (B, K): [[5, 12, 100], [3, 50, 200]]
            2. offsets (B, 1): [[0], [100]]
            3. topk += offsets (Broadcasting):
                [5, 12, 100] + [0] -> [5, 12, 100]
                [3, 50, 200] + [1000] -> [1003, 1050, 1200]
            '''
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # (B, K)
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)  # (P, B, K, A)


            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)  # (P, B, 1, A), (P, B, 1, A)

        return action_mean[0].squeeze(dim=1)  # (B, 1, A) -> (B, A)