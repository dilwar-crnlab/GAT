import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy


from stable_baselines3.common import results_plotter
stable_baselines3.__version__ # printing out stable_baselines version used

from IPython.display import clear_output

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Linear transformations for Query, Key, Value
        self.q_linear = nn.Linear(in_dim, in_dim)
        self.k_linear = nn.Linear(in_dim, in_dim)
        self.v_linear = nn.Linear(in_dim, in_dim)
        
        # Output projection
        self.out = nn.Linear(in_dim, in_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into heads
        q = q.view(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        k = k.view(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        v = v.view(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1)))
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.view(batch_size, -1, x.size(-1))
        
        return self.out(out)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attention = MultiHeadAttention(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        attn_out = self.attention(x)
        out = self.fc(attn_out)
        return self.norm(out)

class GATNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.gat1 = GATLayer(input_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20)  # k_paths * num_bands * 2_fits
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, edge_features, service_info):
        x = self.gat1(edge_features)
        x = F.relu(x)
        x = self.gat2(x)
        
        # Combine with service info
        x = torch.cat([x.mean(dim=1), service_info], dim=1)
        
        # Get action logits and value
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        return action_logits, value
    
# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1, show_plot: bool=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.show_plot = show_plot

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.show_plot and self.n_calls % self.check_freq == 0 and self.n_calls > 5001:
            plotting_average_window = 100

            training_data = pd.read_csv(self.log_dir + 'training.monitor.csv', skiprows=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

            ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')

            ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode service blocking rate')

            ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode bit rate blocking rate')

            # fig.get_size_inches()
            plt.tight_layout()
            plt.show()

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True



class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gat_network = GATNetwork()
    
    def forward(self, obs):
        edge_features = obs['edge_features']
        service_info = obs['service_info']
        return self.gat_network(edge_features, service_info)

def main():
    # Load topology
    topology_name = 'nsfnet_chen_link_span'
    k_paths = 5
    with open(f'../topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
        topology = pickle.load(f)

    # Environment setup
    env_args = dict(
        num_bands=2,
        topology=topology, 
        seed=10,
        allow_rejection=False,
        mean_service_holding_time=5,
        mean_service_inter_arrival_time=0.1,
        k_paths=5,
        episode_length=100
    )

    # Monitor keywords
    monitor_info_keywords = (
        "service_blocking_rate",
        "episode_service_blocking_rate",
        "bit_rate_blocking_rate",
        "episode_bit_rate_blocking_rate"
    )

    # Create directories
    log_dir = "./tmp/deeprmsa-gat/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = gym.make('DeepRMSA-v0', **env_args)
    env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)

    # Initialize PPO with custom GAT policy
    agent = PPO(
        CustomPolicy,
        env,
        verbose=1,
        tensorboard_log="./tb/GAT-DeepRMSA/",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2
    )

    # Training callback
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=log_dir,
        verbose=1
    )

    # Train the agent
    agent.learn(
        total_timesteps=1000000,
        callback=callback
    )

    # Plot results
    plot_training_results(log_dir)

def plot_training_results(log_dir):
    # Load training data
    training_data = pd.read_csv(log_dir + 'training.monitor.csv', skiprows=1)
    
    # Plot metrics
    plotting_average_window = 100
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reward
    rewards = np.convolve(training_data['r'], 
                         np.ones(plotting_average_window)/plotting_average_window, 
                         mode='valid')
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    
    # Service blocking rate
    service_blocking = np.convolve(training_data['episode_service_blocking_rate'],
                                 np.ones(plotting_average_window)/plotting_average_window,
                                 mode='valid')
    ax2.semilogy(service_blocking)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Service Blocking Rate')
    
    # Bit rate blocking rate
    bitrate_blocking = np.convolve(training_data['episode_bit_rate_blocking_rate'],
                                 np.ones(plotting_average_window)/plotting_average_window,
                                 mode='valid')
    ax3.semilogy(bitrate_blocking)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Bit Rate Blocking Rate')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()