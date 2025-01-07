import typing
from dataclasses import dataclass, field
from itertools import islice
from typing import Optional, Sequence, Tuple, Union

import typing
import networkx as nx
import numpy as np

if typing.TYPE_CHECKING:
    from optical_rl_gym.envs.optical_network_env import OpticalNetworkEnv

@dataclass
class PhysicalParameters:
    """Physical layer parameters"""
    h_plank: float = 6.626e-34     
    #alpha: float = 0.046051701859880924,  # 1/km   
    alpha: float = 4.6051701859880924e-2
    gamma: float = 1.2             
    beta_2: float = -21.27e-27     
    beta_3: float = 0.14e-39       
    nse: float = 1.5             
    cr: float = 2.8e-15   #0.028 1/W/km/THz
    launch_power: float =  0.0005011872 #in Watts, P_w= 10 ** ((-3 - 30) / 10) #P_r in Watts, converted from dBm (-3dBm)

@dataclass
class Modulation:
    name: str
    # maximum length in km
    maximum_length: Union[int, float]
    # number of bits per Hz per sec.
    spectral_efficiency: int
    # minimum OSNR that allows it to work
    minimum_osnr: Optional[float] = field(default=None)
    # maximum in-band cross-talk
    inband_xt: Optional[float] = field(default=None)


#
@dataclass
class Span:
    length: float
    attenuation: Optional[float] = field(default=None)
    default_attenuation: Optional[float] = field(default=None)
    noise_figure: Optional[float] = field(default=None)

    def __str__(self):
        return (f"Span(length={self.length}):\n"
                f"\tNodes: {self.node1} -> {self.node2}\n"
                f"\tattenuation: {self.attenuation}km\n"
                f"\tdefault_attenuation: {self.default_attenuation}km\n"
                f"\tnoise_figure: {self.noise_figure}km\n")

               


#
@dataclass
class Link:
    id: int
    length: float
    spans: Tuple[Span, ...]
    node1: Optional[str] = field(default=None)
    node2: Optional[str] = field(default=None)

    def __str__(self):
        spans_str = '\n\t\t'.join([f"Span {i}: length={span.length}km, att={span.attenuation}dB/km, NF={span.noise_figure}dB"
                                  for i, span in enumerate(self.spans)])
        return (f"Link(id={self.id}):\n"
                f"\tNodes: {self.node1} -> {self.node2}\n"
                f"\tLength: {self.length}km\n"
                f"\tSpans:\n\t\t{spans_str}")
#

@dataclass
class Path:
    path_id: int
    node_list: Tuple[str]
    hops: int
    length: Union[int, float]
    links: Tuple[Link, ...]
    idp: Optional[int] = field(default=0)
    best_modulation: Optional[Modulation] = field(default=None)
    current_modulation: Optional[Modulation] = field(default=None)

    def __str__(self):
        # Format node list
        nodes_str = ' -> '.join(self.node_list)
        
        # Format links
        links_str = '\n\t\t'.join([f"Link {i}: {link.node1}->{link.node2}, length={link.length:.2f}km"
                                  for i, link in enumerate(self.links)])
        
        # Format modulations
        best_mod = self.best_modulation.name if self.best_modulation else "None"
        curr_mod = self.current_modulation.name if self.current_modulation else "None"

        return (f"Path(id={self.path_id}):\n"
                f"\tNodes: {nodes_str}\n"
                f"\tHops: {self.hops}\n"
                f"\tTotal Length: {self.length:.2f}km\n"
                f"\tLinks:\n\t\t{links_str}\n"
                f"\tBest Modulation: {best_mod}\n"
                f"\tCurrent Modulation: {curr_mod}")




@dataclass(repr=False)
class Service:
    service_id: int
    source: str
    source_id: int
    destination: Optional[str] = field(default=None)
    destination_id: Optional[str] = field(default=None)
    arrival_time: Optional[float] = field(default=None)
    holding_time: Optional[float] = field(default=None)
    bit_rate: Optional[float] = field(default=None)
    path: Optional[Path] = field(default=None)
    #best_modulation: Optional[Modulation] = field(default=None)
    service_class: Optional[int] = field(default=None)
    number_slots: Optional[int] = field(default=None) # Number of slots used
    #channels: Optional[list] = field(default=None)
    #core: Optional[int] = field(default=None)
    launch_power: Optional[float] = field(default=None) # channel launch power
    accepted: bool = field(default=False)
    #virtual_layer: bool = field(default=False)
    scenario:Optional[int] = None


    # New physical layer attributes
    modulation_format: Optional[str] = None  # e.g., 'QPSK', '16QAM'
    band: Optional[int] = None              # 0 for C-band, 1 for L-band
    initial_slot: Optional[int] = None      # Starting slot
    number_slots: Optional[int] = None      # Number of slots used
    center_frequency: Optional[float] = None # Center frequency of allocation
    termination_time: Optional[float] = None # When service will terminate
    current_OSNR: Optional[float] = None   # in dB
    OSNR_th: Optional[float] = None
    bandwidth: Optional[float] = None #in GHz
    OSNR_margin: Optional[float] =None
    


    def __str__(self):
        # Basic service info
        service_info = [
            f"Service {self.service_id}:",
            f"\tSource: {self.source} (ID: {self.source_id})",
            f"\tDestination: {self.destination} (ID: {self.destination_id})",
            f"\tAccepted: {self.accepted}"
        ]

        # Timing information
        if self.arrival_time is not None:
            service_info.append(f"\tArrival Time: {self.arrival_time:.2f}")
        if self.holding_time is not None:
            service_info.append(f"\tHolding Time: {self.holding_time:.2f}")
        if self.termination_time is not None:
            service_info.append(f"\tTermination Time: {self.termination_time:.2f}")

        # Resource allocation
        if self.bit_rate is not None:
            service_info.append(f"\tBit Rate: {self.bit_rate} Gbps")
        if self.number_slots is not None:
            service_info.append(f"\tNumber of Slots: {self.number_slots}")
        if self.band is not None:
            service_info.append(f"\tBand: {'C' if self.band == 0 else 'L'}")
        if self.initial_slot is not None:
            service_info.append(f"\tInitial Slot: {self.initial_slot}")
        if self.bandwidth is not None:
            service_info.append(f"\tBandwidth: {self.bandwidth:.2f} GHz")

        # Physical layer parameters
        if self.launch_power is not None:
            service_info.append(f"\tLaunch Power: {self.launch_power:.2f} W")
        if self.modulation_format is not None:
            service_info.append(f"\tModulation Format: {self.modulation_format}")
        if self.center_frequency is not None:
            service_info.append(f"\tCenter Frequency: {self.center_frequency:.2f} THz")
        if self.current_OSNR is not None:
            service_info.append(f"\tCurrent OSNR: {self.current_OSNR:.2f} dB")
        if self.OSNR_th is not None:
            service_info.append(f"\tOSNR Threshold: {self.OSNR_th:.2f} dB")
        if self.OSNR_margin is not None:
            service_info.append(f"\tOSNR Margin: {self.OSNR_margin:.2f} dB")

        # Path information if available
        if self.path is not None:
            service_info.append("\tPath:")
            path_str = str(self.path).replace('\n', '\n\t')
            service_info.append(f"\t{path_str}")

        return '\n'.join(service_info)



#Utils added for GNN-DRL approach on 6/1/2025




def start_environment(env: "OpticalNetworkEnv", steps: int) -> "OpticalNetworkEnv":
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight="length"):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


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

def get_best_modulation_format(
    length: float, modulations: Sequence[Modulation]
) -> Modulation:
    # sorts modulation from the most to the least spectrally efficient
    sorted_modulations = sorted(
        modulations, key=lambda x: x.spectral_efficiency, reverse=True
    )
    for i in range(len(modulations)):
        if length <= sorted_modulations[i].maximum_length:
            return sorted_modulations[i]
    raise ValueError(
        "It was not possible to find a suitable MF for a path with {} km".format(length)
    )


def random_policy(env):
    return env.action_space.sample()


def evaluate_heuristic(
    env: "OpticalNetworkEnv",
    heuristic,
    n_eval_episodes=10,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
):
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        _ = env.reset()
        done, _ = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = heuristic(env)
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert (
            mean_reward > reward_threshold
        ), "Mean reward below threshold: " "{:.2f} < {:.2f}".format(
            mean_reward, reward_threshold
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
