
'''
Author: Dilwar Barbhuiya
'''


from math import exp, pi, atan, asinh, log10
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from optical_rl_gym.utils import PhysicalParameters

class OSNRCalculator:
    def __init__(self):
        self.physical_params = PhysicalParameters()

    def calculate_dispersion_params(self, freq_r: float, freq_r_prime: float = None) -> float:
        """Calculate dispersion parameters phi_r or phi_r_r_prime"""
        if freq_r_prime is None:
            return self.physical_params.beta_2 + 2 * pi* self.physical_params.beta_3 * freq_r
        else:
            return (self.physical_params.beta_2 + 
                   pi*self.physical_params.beta_3*(freq_r + freq_r_prime))*(freq_r_prime - freq_r)

    def calculate_ase_noise(self, service, topology) -> float:
        """Calculate ASE noise power considering directed edges"""
        p_ase = 0
        #print("Service in ASE", service)
        
        path = service.path
        #print("Path in ASE", path)

        #print(topology)

        # service.path has links attribute
        constant_term = self.physical_params.nse * self.physical_params.h_plank * service.center_frequency * service.bandwidth
        #print("Const term", constant_term)
        for link in path.links:
            #print("link in ase", link)
            for span in link.spans:
                #print("Span length", span.length)
                #print(type(self.physical_params.alpha), self.physical_params.alpha)
                #pprint(type(span.length), span.length)
                p_ase += constant_term * (exp(self.physical_params.alpha * span.length) - 1)
                    
        return p_ase


    def calculate_sci(self, service, current_node: str, next_node: str, topology) -> float:
        """Calculate SCI for a directed link"""
        link = topology[current_node][next_node]['link']
        #print("Link in SCI", link)
        n_spans = len(link.spans)
        d_l = len([s for s in topology[current_node][next_node]['running_services']])
        #print("D_l", d_l)
        # d_l_reverse = len([s for s in topology[next_node][current_node]['running_services']])
        # print("DL",d_l, d_l_reverse)
        phi_r = self.calculate_dispersion_params(service.center_frequency)
        #print("phi_r", phi_r)
        # print(n_spans)
        
        factor = n_spans * (8/81) * (self.physical_params.gamma**2 * self.physical_params.launch_power**3) / \
                (pi * self.physical_params.alpha**2) * (1/(phi_r * service.bandwidth**2))
        
        const1 = (2 * self.physical_params.alpha - d_l * self.physical_params.launch_power*self.physical_params.cr * service.center_frequency)**2
        
        term1 = ( const1 - (self.physical_params.alpha**2)) / self.physical_params.alpha
        
        term2 = (4 * (self.physical_params.alpha**2) - const1) / (2*self.physical_params.alpha)
        
        p_sci = factor * (
            term1 * asinh((3*pi/(2*self.physical_params.alpha)) * phi_r * (service.bandwidth**2)) +
            term2 * asinh((3*pi/(4*self.physical_params.alpha)) * phi_r * (service.bandwidth**2))
        )
        
        #print("p_sci", p_sci)
        return p_sci

    def calculate_xci(self, service, current_node: str, next_node: str, topology) -> float:
        """Calculate XCI for a directed link"""
        link = topology[current_node][next_node]['link']
        n_spans = len(link.spans)
        #d_l = len([s for s in topology[current_node][next_node]['running_services']])
        
        p_xci = 0
        d_l = len(topology[current_node][next_node]['running_services'])
        if d_l == 0:
            return 0
        #print("Other service in link",  d_l)
        factor = n_spans * (16/81) * ((self.physical_params.gamma**2) * (self.physical_params.launch_power**3)) / (pi**2 * self.physical_params.alpha**2)
        inside_sum = 0
        for other_service in topology[current_node][next_node]['running_services']:
            #print("Running services id", other_service.service_id, "Current service id", service.service_id)
            if other_service.service_id != service.service_id:
                phi_r_r_prime = self.calculate_dispersion_params(service.center_frequency, other_service.center_frequency)
                

                inside_sum_factor = (1/(phi_r_r_prime * other_service.bandwidth))
                #print("Other service bandwidth", other_service.bandwidth)
                #print("Factor in xci", factor)
                
                term1 = ((2*self.physical_params.alpha - d_l*self.physical_params.launch_power*
                        self.physical_params.cr*other_service.center_frequency)**2 - 
                        self.physical_params.alpha**2) / self.physical_params.alpha
                
                term2 = (4*self.physical_params.alpha**2 - 
                        (2*self.physical_params.alpha - d_l*self.physical_params.launch_power*
                        self.physical_params.cr*other_service.center_frequency)**2) / \
                        (2*self.physical_params.alpha)
                
                
                inside_sum  += inside_sum_factor * (term1 * atan((2*pi**2/self.physical_params.alpha) * phi_r_r_prime * service.bandwidth) + (term2 * 
                                            atan((pi**2/self.physical_params.alpha) * phi_r_r_prime * service.bandwidth)))
        p_xci = inside_sum * factor
        #print("p_xci", p_xci)  
        return p_xci

    def calculate_osnr(self, service, topology) -> float:
        """Calculate OSNR considering directed path"""
        # Calculate ASE noise
        #print("Service in OSNR method", service)
        p_ase = self.calculate_ase_noise(service, topology)
        #print("P_ase", p_ase)
        
        # Calculate total NLI noise
        p_nli = 0
        path = service.path.node_list
        
        # Iterate through directed path
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Calculate SCI and XCI for this directed edge
            p_nli += self.calculate_sci(service, current_node, next_node, topology)
            p_nli += self.calculate_xci(service, current_node, next_node, topology)
            
        # Calculate OSNR
        
        osnr = self.physical_params.launch_power / (p_ase + p_nli)
        #print("osnr", osnr)
        
        # Convert to dB
        osnr_db = 10 * log10(osnr)
        #print("osnr_db", osnr_db)
        
        return osnr_db