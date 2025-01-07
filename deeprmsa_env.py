from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv



from .optical_network_env import OpticalNetworkEnv

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        num_bands,
        topology=None,
        j=2,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        k_paths=5,
        allow_rejection=False,
    ):
        super().__init__(
            num_bands=num_bands,
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            k_paths=k_paths,
            allow_rejection=allow_rejection,
            reset=False,
        )

        self.j = j

        # New action space for (path, band, fit) selection
        total_actions = self.k_paths * self.num_bands * self.j + self.reject_action # 2 fits per band
        self.action_space = gym.spaces.Discrete(total_actions)

        max_path_length = self.get_max_path_length() #max number of edges in a path
        # New observation space for GAT features
        self.observation_space = gym.spaces.Dict({
            'edge_features': gym.spaces.Box(
                low=-1, high=1, 
                shape=(self.k_paths, max_path_length, 16),  # 16 edge features
                dtype=np.float32
            ),
            'service_info': gym.spaces.Box(
                low=0, high=1,
                shape=(2,),  # bit_rate, holding_time
                dtype=np.float32
            )
        })
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters=False)

    def get_max_path_length(self):
        """
        Computes maximum number of edges in any path from k-shortest paths.
        """
        max_length = 0
        
        # Check all source-destination pairs
        for source in self.topology.nodes():
            for dest in self.topology.nodes():
                if source != dest:
                    # Get k-shortest paths for this pair
                    paths = self.k_shortest_paths[source, dest]
                    
                    # Check length of each path
                    for path in paths:
                        num_edges = len(path.node_list) - 1  # number of edges = nodes - 1
                        max_length = max(max_length, num_edges)
                        
        return max_length

    def observation(self):
        """Returns GAT-ready observation."""
        # Get edge features for all k-paths
        path_features = self.compute_path_edge_features(self.current_service)
        
        # Service information
        service_info = np.array([
            self.current_service.bit_rate / 100,
            self.current_service.source_id, 
            self.current_service.destination_id
        ])
        
        return {
            'edge_features': path_features,
            'service_info': service_info
        }

    def step(self, action):
        """Handle GAT-based action selection."""
        path_idx, band, fit_type = self._get_route_band_block_id(action)
        
        # Get blocks for selected band
        blocks = self.get_available_blocks(path_idx, self.num_bands, band, self.modulations)
        
        # Select block based on fit_type
        if fit_type == 0:  # first-fit
            initial_slot = blocks['first_fit'][0][0] if blocks['first_fit'][0].size > 0 else -1
        else:  # last-fit
            initial_slot = blocks['last_fit'][0][0] if blocks['last_fit'][0].size > 0 else -1
        
        if initial_slot >= 0:
            # Try to provision
            result = super().step([path_idx, band, initial_slot])
        else:
            # Reject if no valid block
            result = super().step([self.k_paths, self.num_bands, self.num_spectrum_resources])

        obs, rw, _, info = result
        
        return result



    
    def reward(self, path_idx, band, fit_type):
        """
        Compute reward considering:
        1. Service acceptance
        2. OSNR margin of allocation
        3. Fragmentation impact
        4. Resource utilization
        """
        if not self.current_service.accepted:
            return -1
            
        # Get selected path
        path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path_idx]
        reward = 0
        
        # 1. Base reward for acceptance
        reward += 1
        
        # 2. OSNR margin reward
        osnr_margin = self.current_service.OSNR_margin
        osnr_th = self.current_service.OSNR_th
        normalized_osnr = osnr_margin / osnr_th
        reward += 0.3 * normalized_osnr
        
        # 3. Fragmentation penalty
        total_fragmentation = 0
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            if band == 0:  # C-band
                frag = self.topology[node1][node2]['c_band_fragmentation']
            else:  # L-band
                frag = self.topology[node1][node2]['l_band_fragmentation']
            total_fragmentation += frag
        
        avg_fragmentation = total_fragmentation / (len(path.node_list) - 1)
        reward -= 0.2 * avg_fragmentation  # penalty for fragmentation
        
        # 4. Utilization balance reward
        total_utilization = 0
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            if band == 0:
                util = self.topology[node1][node2]['c_band_util']
            else:
                util = self.topology[node1][node2]['l_band_util']
            total_utilization += util
        
        avg_utilization = total_utilization / (len(path.node_list) - 1)
        # Reward for balanced utilization (closest to 0.5)
        balance_metric = 1 - 2 * abs(0.5 - avg_utilization)
        reward += 0.2 * balance_metric

        # Band preference based on modulation
        modulation = self.current_service.modulation_format
        if modulation in ['BPSK', 'QPSK']:
            if band == 1:  # L-band
                reward += 0.2  # Bonus for using preferred band
            else:  # C-band
                reward -= 0.1  # Penalty for using non-preferred band
        elif modulation in ['8QAM', '16QAM']:
            if band == 0:  # C-band
                reward += 0.2  # Bonus for using preferred band
            else:  # L-band
                reward -= 0.1  # Penalty for using non-preferred band
        
        return reward

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)


    
    def _get_route_band_block_id(self, action):
        path_index = action // 4  # 4 combinations per path
        remaining = action % 4
        band = remaining // 2     # 2 fits per band
        fit_type = remaining % 2  # first/last
        return path_index, band, fit_type


