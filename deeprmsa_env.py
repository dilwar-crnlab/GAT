from typing import Tuple

import gym
import gym.spaces
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
        #print("Max path length",max_path_length)
        # New observation space for GAT features
        num_nodes = self.topology.number_of_nodes()
    

        num_edge_features = self.k_paths * self.get_max_path_length() * 16
        num_service_info = 2 * self.topology.number_of_nodes() + 1  # Source-destination + bit rate

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_edge_features + num_service_info,),
            dtype=np.float32
        )
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
        """Returns a flattened observation combining edge features and service info."""
        # Get edge features (shape: [k_paths, max_path_length, 16])
        path_features = self.compute_path_edge_features(self.current_service)
        
        # Create edge features array
        max_path_length = self.get_max_path_length()
        edge_features_array = np.zeros((self.k_paths, max_path_length, 16), dtype=np.float32)
        for path_idx, features in path_features.items():
            path_len = len(features)
            if path_len > max_path_length:
                edge_features_array[path_idx] = features[:max_path_length]
            else:
                edge_features_array[path_idx, :path_len] = features
        
        # Flatten edge features
        flat_edge_features = edge_features_array.flatten()

        # Create source-destination one-hot encoding
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()), dtype=np.float32)
        source_destination_tau[0, self.current_service.source_id] = 1
        source_destination_tau[1, self.current_service.destination_id] = 1

        # Flatten source-destination
        flat_source_destination = source_destination_tau.flatten()

        # Normalize bit rate
        bit_rate = np.array([self.current_service.bit_rate / 100], dtype=np.float32)

        # Combine all parts into a single flattened array
        flattened_observation = np.concatenate([flat_edge_features, flat_source_destination, bit_rate])

        #print(f"Flattened observation shape: {flattened_observation.shape}")
        #print(f"Expected shape: {self.observation_space.shape}")
        
        return flattened_observation


    def step(self, action):
        """Handle GAT-based action selection."""
        path_idx, band, fit_type = self._get_route_band_block_id(action)
        #print(path_idx, band, fit_type)
        
        # Get blocks for selected band
        blocks = self.get_available_blocks_FLF(path_idx, self.num_bands, band, self.modulations)
        #print(blocks)
        
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



    
    def reward(self, path_idx, band):
        return super().reward(path_idx, band)
        

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)


    
    def _get_route_band_block_id(self, action):
        path_index = action // 4  # 4 combinations per path
        remaining = action % 4
        band = remaining // 2     # 2 fits per band
        fit_type = remaining % 2  # first/last
        return path_index, band, fit_type


