import gym
import time
import numpy as np
from tqdm import tqdm

env = gym.make('MountainCar-v0')

# # reset to starting State
# s = env.reset()
# action_list = list(range(env.action_space.n))
# while True:
#     a = np.random.choice(action_list)   # random an action
#     s, r, done, _ = env.step(a)      # perform action, get state, reward, is terminated
#     if done:
#         break
#     env.render()
#     time.sleep(0.05)


class NBinDiscretizer:
    def __init__(self, min_val, max_val, nbins):
        self.min_val = min_val
        self.max_val = max_val
        self.step = (max_val - min_val) / nbins
        self.nbins = int(nbins)
    def __call__(self, val):
        return int(round((val - self.min_val) / self.step)) % self.nbins


class Dicretezation:
    def __init__(self, discretezers):
        self.discretezers = discretezers
    def __getitem__(self, index):
        assert len(index) == len(self.discretezers)
        return tuple([self.discretezers[i](index[i]) for i in range(len(index))])


lr = 0.1
gamma = 0.9
n_quantization = 50
x_quantizer = NBinDiscretizer(env.observation_space.low[0], env.observation_space.high[0], n_quantization)
v_quantizer = NBinDiscretizer(env.observation_space.low[0], env.observation_space.high[0], n_quantization)
state_quantizer = Dicretezation([x_quantizer, v_quantizer])
Q = np.zeros((n_quantization, n_quantization, 3))
# inititalize some variables
epochs = 10000
epsilon = 0.9
epsilon_scale = epsilon / (epochs / 4)

# some metrics
max_reward = -1000
max_pos = -1000
progress_bar = tqdm(range(epochs), desc="Epoch", unit='epoch', position=0, dynamic_ncols=True, bar_format="{desc}: {n_fmt}/{total_fmt} - {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
for epoch in progress_bar:
    ep_max_pos = -1000
    ep_reward = 0
    
    # reset environment
    obs = env.reset()
    done = False

    while not done:
        
        # take an action
        if np.random.random_sample() > epsilon:
            a = np.argmax(Q[state_quantizer[obs]])
        else:
            a = np.random.randint(0, env.action_space.n)
        
        # perform action
        new_obs, r, done, info = env.step(a)
        ep_reward += r
        
        if new_obs[0] >= env.goal_position:
            print(f"\nReach goal at epoch {epoch} with reward: {ep_reward}")
        # update Q
        cur_q_value = Q[state_quantizer[obs]][a]        
        new_q_value = (1-lr) * cur_q_value + lr * (r + gamma * max(Q[state_quantizer[new_obs]]))
        Q[state_quantizer[obs]][a] = new_q_value
        obs = new_obs
        ep_max_pos = max(obs[0], ep_max_pos)
        
    max_reward = max(ep_reward, max_reward)
    max_pos = max(ep_max_pos, max_pos)
    epsilon = max(0, epsilon - epsilon_scale)
    
    progress_bar.set_postfix(dict(epoch=epoch, ep_reward=ep_reward, max_reward=max_reward, ep_max_pos=ep_max_pos, max_pos=max_pos, epsilon=epsilon))