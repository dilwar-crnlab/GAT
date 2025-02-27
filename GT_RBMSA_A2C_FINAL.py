import os
import json
import gym
import math
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Union, Type, Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class GATLayer(nn.Module):
    def __init__(self, output_dim: int, input_dim: int, num_heads: int = 8, dropout_rate: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.W = nn.Parameter(th.empty(num_heads, input_dim, output_dim))
        # self.a: [num_heads, 2 * output_dim, 1]
        self.a = nn.Parameter(th.empty(num_heads, 2 * output_dim, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.a, gain=gain)

    def forward(self, inputs):
        H, adj = inputs  # H: [B, N, input_dim], adj: [B, N, N]
        B, N, _ = H.shape
        Wh = th.einsum('bni,hio->bhno', H, self.W)
        Wh_i = Wh.unsqueeze(3)
        Wh_j = Wh.unsqueeze(2)
        
        # Concatenate along the feature dimension: [B, num_heads, N, N, 2*output_dim]
        concat = th.cat([Wh_i.expand(-1, -1, -1, N, -1),
                         Wh_j.expand(-1, -1, N, -1, -1)], dim=-1)
        a_exp = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        e = th.matmul(concat.unsqueeze(-2), a_exp).squeeze(-1).squeeze(-1)
        
        # Apply mask: adj: [B, N, N] -> unsqueeze to [B, 1, N, N]
        mask = adj.unsqueeze(1).bool()
        e = th.where(mask, e, th.tensor(-9e15, device=e.device))
        
        # Softmax to obtain attention coefficients over neighbors.
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        h_prime = th.matmul(alpha, Wh)
        h_prime = h_prime.permute(0, 2, 1, 3).reshape(B, N, self.num_heads * self.output_dim)
        
        return F.relu(h_prime), alpha

class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, learnable_pos: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.learnable_pos = learnable_pos
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding: either learnable or fixed (using sinusoidal encoding)
        if learnable_pos:
            self.pos_embedding = nn.Parameter(th.randn(max_seq_length, d_model) * 0.1)
        else:
            pos_encoding = self._get_sinusoidal_encoding(max_seq_length, d_model)
            self.register_buffer('pos_embedding', pos_encoding)
            
    def _get_sinusoidal_encoding(self, max_seq_length: int, d_model: int):
        pos = th.arange(max_seq_length).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = th.zeros(max_seq_length, d_model)
        pos_encoding[:, 0::2] = th.sin(pos * div_term)
        pos_encoding[:, 1::2] = th.cos(pos * div_term[:d_model//2])
        return pos_encoding

    def forward(self, inputs):
        batch_size, k_paths, seq_length = inputs.shape

        # Flatten the first two dimensions to feed into the embedding layer.
        flat_inputs = inputs.reshape(-1, seq_length)  # shape: [batch_size * k_paths, seq_length]
        token_embeds = self.token_embedding(flat_inputs)  # shape: [batch_size * k_paths, seq_length, d_model]
        
        # Get positional encoding for the given sequence length.
        # Instead of calling expand(), rely on broadcasting.
        pos_encoding = self.pos_embedding[:seq_length].unsqueeze(0)  # shape: [1, seq_length, d_model]
        output = token_embeds + pos_encoding  # shape: [batch_size * k_paths, seq_length, d_model]
        
        # Reshape back to the original dimensions
        output = output.reshape(batch_size, k_paths, seq_length, self.d_model)
        return output

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, path_inputs, gat_kv, training=True, self_attn_mask=None):
        batch_size_k = path_inputs.size(0)
        seq_length = path_inputs.size(1)
        gat_kv = gat_kv.repeat_interleave(batch_size_k // gat_kv.size(0), dim=0) 
        # Transpose for attention: [seq_length, batch_size * k_paths, d_model]
        path_inputs = path_inputs.transpose(0, 1)
        gat_kv = gat_kv.transpose(0, 1)
        
        if self_attn_mask is not None:
            self_attn_mask = ~self_attn_mask
            #print(f"Final mask shape: {self_attn_mask.shape}")

        # Self-attention
        attn_output1, _ = self.self_attention(
            path_inputs, path_inputs, path_inputs,
            key_padding_mask=self_attn_mask,
            need_weights=False
        )
        attn_output1 = self.dropout(attn_output1)
        out1 = self.norm1(path_inputs + attn_output1)
        
        # Cross-attention
        attn_output2, _ = self.cross_attention(
            query=out1,
            key=gat_kv,
            value=gat_kv,
            need_weights=False
        )
        attn_output2 = self.dropout(attn_output2)
        out2 = self.norm2(out1 + attn_output2)
        
        # Feed-forward
        out2 = out2.transpose(0, 1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.norm3(out2 + ffn_output)
        
        return out3

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, inputs, mask=None):
        scores = self.attention(inputs).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            
        weights = F.softmax(scores, dim=-1)
        weights = weights.unsqueeze(-1)
        
        pooled = (inputs * weights).sum(dim=-2)
        return pooled

class CandidatePathProcessor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, 
                 num_heads: int, ff_dim: int, learnable_pos: bool = False, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length    
        self.embedding_layer = EmbeddingWithPositionalEncoding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_length=max_seq_length,
            learnable_pos=learnable_pos
        )
        
        self.transformer_block = SimpleTransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )
        
        self.attention_pooling = AttentionPooling(d_model)
        
    def forward(self, candidate_tokens, gat_kv, training=True, mask=None):

        # Embed tokens
        embedded = self.embedding_layer(candidate_tokens)
        # Reshape for transformer
        batch_size, k_paths, seq_length, d_model = embedded.shape
        embedded_flat = embedded.view(-1, seq_length, d_model)
        # Prepare mask
        if mask is not None:
            mask = mask.view(-1, seq_length)
        # Process through transformer
        transformer_out = self.transformer_block(
            embedded_flat, 
            gat_kv,
            training=training,
            self_attn_mask=mask
        )
        # Reshape back
        transformer_out = transformer_out.view(batch_size, k_paths, seq_length, d_model)
        # Pool output
        pooled_output = self.attention_pooling(transformer_out, mask=mask.view(batch_size, k_paths, seq_length))
        #print(f"Pooled output shape: {pooled_output.shape}")
        
        return pooled_output
    


class DeepRMSANetwork(nn.Module):
    def __init__(
        self,
        num_original_nodes: int,
        num_edges: int,
        k_paths: int,
        num_bands: int,
        P: int,
        hidden_size: int = 128,
        num_actions: int = None,
        num_dense_layers: int = 4,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        gat_dropout: float = 0.2,
        transformer_num_heads: int = 8,
        transformer_ff_dim: int = 256
    ):
        super().__init__()
        super().__init__()
        #print(f"P value in init: {P}")
        self.num_original_nodes = num_original_nodes
        self.num_edges = num_edges
        self.k_paths = k_paths
        self.num_bands = num_bands
        self.hidden_size = hidden_size
        self.P = P
        self.d_model = hidden_size


        # Calculate input dimension for GAT
        features_per_band = 8
        general_link_features = 1
        features_per_link = general_link_features + (features_per_band * num_bands)

        # Calculate input dimension for dense layers
        self.feature_dim = hidden_size + 6 + (2 * num_original_nodes + 1)  # d_model + spectrum_features + global_features
        
        # GAT Layer
        # GAT Layer with correct input dimension
        self.gat = GATLayer(
            input_dim=features_per_link,  # Input features per link
            output_dim=hidden_size // num_heads,  # Output dimension per head
            num_heads=num_heads,
            dropout_rate=gat_dropout
        )
        
        # Transformer for path processing
        self.processor = CandidatePathProcessor(
            vocab_size=num_edges,
            d_model=hidden_size,
            max_seq_length=P,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            learnable_pos=False,
            dropout_rate=dropout_rate
        )

        # Dense layers with correct dimensions
        self.dense_layers = nn.ModuleList([
            nn.Linear(self.feature_dim, hidden_size),  # First layer maps to hidden_size
            *[nn.Linear(hidden_size, hidden_size) for _ in range(num_dense_layers - 1)],  # Subsequent layers
            nn.Dropout(dropout_rate)
        ])

    def forward(self, inputs):
        batch_size = inputs.size(0)
        idx = 0

        #print("Inputes", inputs.shape)

        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
    
        batch_size = inputs.size(0)
        idx = 0
        
        # Extract state components
        source_dest_size = 2 * self.num_original_nodes
        source_dest = inputs[:, idx:idx + source_dest_size]
        idx += source_dest_size
        
        slots_path_size = self.k_paths
        slots_per_path = inputs[:, idx:idx + slots_path_size]
        idx += slots_path_size
        
        spectrum_size = self.k_paths * 6 * self.num_bands
        spectrum = inputs[:, idx:idx + spectrum_size]
        spectrum = spectrum.view(batch_size, self.k_paths, self.num_bands, 6)
        idx += spectrum_size
        
        # Link features
        features_per_band = 8
        general_link_features = 1
        features_per_link = general_link_features + (features_per_band * self.num_bands)
        link_features_size = self.num_edges * features_per_link
        
        link_features = inputs[:, idx:idx + link_features_size]
        link_features = link_features.view(batch_size, self.num_edges, features_per_link)
        idx += link_features_size
        
        # Edge adjacency
        edge_adj_size = self.num_edges * self.num_edges
        edge_adj = inputs[:, idx:idx + edge_adj_size]
        edge_adj = edge_adj.view(batch_size, self.num_edges, self.num_edges)
        idx += edge_adj_size

        #print(f"Edge adjacency shape: {edge_adj.shape}")
        
        # Path vector
        path_vector_size = self.num_edges * self.k_paths
        path_vector = inputs[:, idx:idx + path_vector_size]
        path_vector = path_vector.view(batch_size, self.num_edges, self.k_paths)
        
        # Process through GAT
        gat_output, _ = self.gat([link_features, edge_adj])
        
        # Process paths
        padded_tokens, mask = self.tokenize_candidate_paths(path_vector)
        attn_pooled_path = self.processor(padded_tokens, gat_output, mask=mask)
        
        # Process global features
        source_dest_tiled = source_dest.unsqueeze(1).repeat(1, self.k_paths, 1)
        slots_expanded = slots_per_path.unsqueeze(-1)
        global_features = th.cat([source_dest_tiled, slots_expanded], dim=-1)

        
        # Process C and L bands
        shared_c = self._process_band(
            attn_pooled_path,
            spectrum[:, :, 0, :],  # C-band features
            global_features
        )
        shared_l = self._process_band(
            attn_pooled_path,
            spectrum[:, :, 1, :],  # L-band features
            global_features
        )
        
        # Combine features
        shared_features = th.cat([shared_c, shared_l], dim=-1)
        #print(f"Final shared features shape: {shared_features.shape}")
        
        return shared_features

    def _process_band(self, path_features, band_spectrum, global_features):

        x = th.cat([path_features, band_spectrum, global_features], dim=-1)


        # Process through dense layers
        batch_size, k_paths, _ = x.shape
        x = x.view(batch_size * k_paths, -1)  # Flatten for dense layers

        
        for layer in self.dense_layers:
            x = F.relu(layer(x))    
        x = x.view(batch_size, k_paths, -1)  # Reshape back
        return x

    def tokenize_candidate_paths(self, path_vector):
        batch_size = path_vector.size(0)
        num_edges = path_vector.size(1)
        k_paths = path_vector.size(2)
        
        def tokenize_single_candidate(candidate):
            indices = th.where(candidate == 1.0)[0]
            num_tokens = indices.size(0)
            pad_length = max(0, self.P - num_tokens)
            padded = F.pad(indices, (0, pad_length), value=0)
            return padded[:self.P]
        
        # Process each path
        tokenized = []
        for i in range(batch_size):
            for j in range(k_paths):
                tokenized.append(tokenize_single_candidate(path_vector[i, :, j]))
        
        tokenized = th.stack(tokenized).view(batch_size, k_paths, self.P)
        mask = tokenized != 0

        return tokenized, mask
    

class AC_Net(nn.Module):
    def __init__(self, scope, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar,
                 num_original_nodes, num_edges, k_paths, num_bands, P, network_params=None):
        super().__init__()
        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar

        # Default network parameters if none provided
        if network_params is None:
            network_params = {
                'num_dense_layers': 4,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'gat_dropout': 0.2,
                'transformer_num_heads': 8,
                'transformer_ff_dim': 256
            }
        # Shared feature extractor
        self.feature_extractor = DeepRMSANetwork(
            num_original_nodes=num_original_nodes,
            num_edges=num_edges,
            k_paths=k_paths,
            num_bands=num_bands,
            P=P,
            hidden_size=layer_size,
            num_actions=n_actions,
            num_dense_layers=network_params['num_dense_layers'],
            num_heads=network_params['num_heads'],
            dropout_rate=network_params['dropout_rate'],
            gat_dropout=network_params['gat_dropout'],
            transformer_num_heads=network_params['transformer_num_heads'],
            transformer_ff_dim=network_params['transformer_ff_dim']
        )

        self.policy_head = nn.Sequential(
            nn.Linear(k_paths * layer_size * 2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, n_actions),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(k_paths * layer_size * 2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
        )

    def forward(self, x):
        shared_features = self.feature_extractor(x)  # [batch_size, k_paths, feature_dim]
        
        flat_features = shared_features.view(shared_features.size(0), -1)  # [batch_size, k_paths * (layer_size*2)]
        # Process through policy head
        policy_logits = self.policy_head(flat_features)
        value = self.value_head(flat_features)

        
        return policy_logits, value

class DeepRMSA_A2C:
    def __init__(self, env, num_layers=4, layer_size=128, learning_rate=1e-4, regu_scalar=1e-3,
                 num_dense_layers=4,
        num_heads=8,
        dropout_rate=0.1,
        gat_dropout=0.2,
        transformer_num_heads=8,
        transformer_ff_dim=256):
        self.env = env
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.x_dim = self.calculate_state_dim(env)
        self.P = self.find_longest_path(env.topology.graph["ksp"])

        # Initialize network
        self.network = AC_Net(
            scope='a2c',
            x_dim_p=self.x_dim,
            x_dim_v=self.x_dim,
            n_actions=env.action_space.n,
            num_layers=num_layers,
            layer_size=layer_size,
            regu_scalar=regu_scalar,
            num_original_nodes=env.topology.number_of_nodes(),
            num_edges=env.topology.number_of_edges(),
            k_paths=env.k_paths,
            num_bands=env.num_bands,
            P=self.P,
            # Pass network hyperparameters
            network_params={
                'num_dense_layers': num_dense_layers,
                'num_heads': num_heads,
                'dropout_rate': dropout_rate,
                'gat_dropout': gat_dropout,
                'transformer_num_heads': transformer_num_heads,
                'transformer_ff_dim': transformer_ff_dim
            }
        ).to(self.device)

        # Optimizer
        self.optimizer = th.optim.Adam(self.network.parameters(), lr=learning_rate)

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_blocking_rates = []
        self.episode_bit_rate_blocking_rates = []
        self.episode_count = 0


    @th.no_grad()
    def choose_action(self, state):
        state_tensor = self._to_tensor(state)
        policy, _ = self.network(state_tensor)
        
        # Make sure policy is properly shaped [batch_size, n_actions]
        if len(policy.shape) == 1:
            policy = policy.unsqueeze(0)
            
        # Sample action
        try:
            action = th.multinomial(policy, 1).squeeze()
            return action.item() if isinstance(action.item(), int) else action[0].item()
        except Exception as e:
            # print(f"Policy shape: {policy.shape}")
            # print(f"Policy values: {policy}")
            raise e

    def calculate_state_dim(self, env):
        features_per_band = 8
        general_link_features = 1
        features_per_link = general_link_features + (features_per_band * env.num_bands)
        link_features_size = env.topology.number_of_edges() * features_per_link
        
        return (2 * env.topology.number_of_nodes() + 
                env.k_paths +
                (env.k_paths * 6 * env.num_bands) +
                link_features_size +
                (env.topology.number_of_edges() * env.topology.number_of_edges()) +
                (env.topology.number_of_edges() * env.k_paths))

    def find_longest_path(self, k_shortest_paths):
        max_hops = 0
        for paths in k_shortest_paths.values():
            for path in paths:
                max_hops = max(max_hops, len(path.node_list) - 1)
        return max_hops

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return th.FloatTensor(x).to(self.device)
        return x
    

    


    def train(self, total_timesteps: int, n_steps: int, gamma: float):
        episode_count = 0
        total_steps = 0
        state = self.env.reset()
        episode_reward = 0

        pbar = tqdm(total=self.env.episode_length, desc="Episode Progress")
        
        while total_steps < total_timesteps:
            values = []
            states = []
            actions = []
            rewards = []
            dones = []
            
            # Collect n-step experiences
            for _ in range(n_steps):
                state_tensor = self._to_tensor(state)
                with th.no_grad():
                    policy, value = self.network(state_tensor)
                    action = th.multinomial(policy, 1).item()
                
                next_state, reward, done, info = self.env.step(action)
                
                states.append(state_tensor)
                actions.append(action)
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                total_steps += 1
                pbar.update(1)

                if done:
                    episode_count += 1
                    self.episode_count += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_blocking_rates.append(info.get('episode_service_blocking_rate', 0))
                    self.episode_bit_rate_blocking_rates.append(info.get('episode_bit_rate_blocking_rate', 0))
                    
                    state = self.env.reset()
                    episode_reward = 0
                    pbar.reset()

            # Compute returns and advantages
            state_tensor = self._to_tensor(state)
            with th.no_grad():
                policy, value = self.network(state_tensor)
                next_value = value if not done else th.tensor(0.0).to(self.device)
                #print(f"Train loop - Policy shape: {policy.shape}, values: {policy}")
                action = self.choose_action(state)  # Use choose_action method


            returns = self._compute_returns(rewards, next_value, dones, gamma)
            returns = th.tensor(returns, dtype=th.float32, device=self.device)
            values = th.cat([v for v in values])
            advantages = returns - values.squeeze()

            # Update policy
            states = th.stack(states)
            actions = th.tensor(actions, device=self.device)
            
            policy_loss, value_loss, entropy = self._update_policy(
                states=states,
                actions=actions,
                returns=returns,
                advantages=advantages
            )

            # Log metrics
            #if episode_count % 10 == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            mean_blocking = np.mean(self.episode_blocking_rates[-100:]) if self.episode_blocking_rates else 0
            
            pbar.set_postfix({
                'episode': self.episode_count,
                'mean_reward': f'{mean_reward:.2f}',
                'mean_blocking': f'{mean_blocking:.4f}',
                'policy_loss': f'{policy_loss:.4f}',
                'value_loss': f'{value_loss:.4f}',
                'entropy': f'{entropy:.4f}'
            })

        pbar.close()

    def _compute_returns(self, rewards, next_value, dones, gamma):
        returns = []
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def _update_policy(self, states, actions, returns, advantages):
        # Get policy and value predictions
        policy, value = self.network(states)
        
        # Compute policy loss
        log_prob = th.log(policy + 1e-10)
        action_log_probs = log_prob.gather(1, actions.unsqueeze(1))
        entropy = -(policy * log_prob).sum(-1).mean()
        policy_loss = -(action_log_probs * advantages.unsqueeze(1)).mean()
        policy_loss = policy_loss - 0.01 * entropy  # entropy bonus

        # Compute value loss
        value_loss = 0.5 * (returns - value.squeeze()).pow(2).mean()

        # Total loss
        loss = policy_loss + value_loss

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
    





def main():
    parser = argparse.ArgumentParser(description='Train or retrain DeepRMSA A2C')
    parser.add_argument('--timesteps', type=int, default=15000,
                        help='Total timesteps for training')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--load_model', type=str, default="",
                        help='Path to a saved checkpoint to retrain from')
    args = parser.parse_args()
    
    # Set random seeds
    th.manual_seed(42)
    np.random.seed(42)
    # Set random seeds
    th.manual_seed(42)
    np.random.seed(42)
    

    # Set up device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if th.cuda.is_available():
        th.cuda.manual_seed(42)
    
    # Load topology
    topology_name = 'cost256'
    k_paths = 5
    with open(f'../topologies/{topology_name}_{k_paths}-paths_new.h5', 'rb') as f:
        topology = pickle.load(f)
    IAT = 0.067
    
    # Environment setup
    env_args = dict(
        num_bands=2,
        topology=topology,
        seed=10,
        allow_rejection=False,
        j=1,
        mean_service_holding_time=10.0,
        mean_service_inter_arrival_time=IAT,
        k_paths=k_paths,
        episode_length=1500,
        node_request_probabilities=None
    )
    
    # Create directories with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"./logs/deeprmsa-a2c/{IAT}/{current_time}"
    model_dir = f"./models/deeprmsa-a2c/{IAT}/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Starting training...")
    print(f"Logging to {log_dir}")
    print(f"Models will be saved to {model_dir}")
    training_params = {'total_timesteps': args.timesteps, 'gamma': args.gamma}
    for key, value in training_params.items():
        print(f"{key}: {value}")
    for key, value in env_args.items():
        print(f"{key}: {value}")
    
    # Create the environment
    env = gym.make('DeepRMSA-v0', **env_args)
    
    # Create A2C agent
        #Best hyperparameters:

    num_gat_heads= 8
    transformer_num_heads=4
    hidden_size= 176
    t_num_layers= 2
    learning_rate= 5.8e-05
    regu_scalar= 0.000111
    n_steps= 10
    gamma= 0.91
    num_dense_layers= 2
    transformer_ff_dim= 256
    dropout_rate= 0.42
    gat_dropout= 0.19


# Create and train model
    agent = DeepRMSA_A2C(
        env=env,
        num_layers=t_num_layers,
        layer_size=hidden_size,  # Use hidden_size for layer_size
        learning_rate=learning_rate,
        regu_scalar=regu_scalar,
        num_dense_layers=num_dense_layers,
        num_heads=num_gat_heads,
        dropout_rate=dropout_rate,
        gat_dropout=gat_dropout,
        transformer_num_heads=transformer_num_heads,
        transformer_ff_dim=transformer_ff_dim
    )
    
    # Optionally load a saved checkpoint to continue training
    if args.load_model and os.path.isfile(args.load_model):
        checkpoint = th.load(args.load_model)
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.episode_rewards = checkpoint.get('episode_rewards', [])
        agent.episode_lengths = checkpoint.get('episode_lengths', [])
        agent.episode_blocking_rates = checkpoint.get('episode_blocking_rates', [])
        agent.episode_count = checkpoint.get('episode_count', 0)
        print(f"Loaded model and metrics from {args.load_model}")
    
    with open(os.path.join(log_dir, 'training_params.txt'), 'w') as f:
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
    
    try:
        # Train the agent
        agent.train(
            total_timesteps=training_params['total_timesteps'],
            gamma=training_params['gamma'],
            n_steps=10
        )
        
        final_model_path = os.path.join(model_dir, 'final_model.pth')
        # Save checkpoint including metrics
        checkpoint = {
            'model_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode_rewards': agent.episode_rewards,
            'episode_lengths': agent.episode_lengths,
            'episode_blocking_rates': agent.episode_blocking_rates,
            'episode_count': agent.episode_count
        }
        th.save(checkpoint, final_model_path)
        print(f"\nTraining completed. Final checkpoint saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupted_model_path = os.path.join(model_dir, 'interrupted_model.pth')
        checkpoint = {
            'model_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode_rewards': agent.episode_rewards,
            'episode_lengths': agent.episode_lengths,
            'episode_blocking_rates': agent.episode_blocking_rates,
            'episode_count': agent.episode_count
        }
        th.save(checkpoint, interrupted_model_path)
        print(f"Interrupted checkpoint saved to {interrupted_model_path}")
        
    finally:
        training_data = {
            'rewards': agent.episode_rewards,
            'lengths': agent.episode_lengths,
            'blocking_rates': agent.episode_blocking_rates,
            'episode_count' : agent.episode_count
        }
        np.save(os.path.join(log_dir, 'training_data.npy'), training_data)
        env.close()

if __name__ == "__main__":
    main()
