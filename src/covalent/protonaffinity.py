"""  
Proton affinity of the α-carbon also correlates because both reflect
the intrinsic electrophilicity / LUMO character
    
https://www.ks.uiuc.edu/Training/SumSchool/materials/sources/tutorials/05-qm-tutorial/part2.html

The proton affinity for the reaction is defined as the negative of the reaction enthalpy at 298.15K, 
and hence (T: temperature, R: ideal gas constant)

    P(A) = -ΔH = -ΔE + RT
    
The energy of a nonlinear polyatomic molecule can be approximated as
    
    E(T) = E_rot(T) + E_trans(T) + ZPE + E'_vib(T) + E_ele

Here ZPE stands for the zero point energy of the normal modes. From statistical mechanics we know 
that the contributions E_rot and E_trans both equal 3/2 RT. Furthermore, E'_vib can usually be 
neglected compared to the zero point energy ZPE. 

To determine the proton affinity one has to calculate the energy change in going 
from the reactant A- and H+ to the product AH. The rotational energy contribution remains 
constant (since the proton does not posses rotational kinetic energy) and the translational 
energy of the proton contributes -3/2 RT. 

    For A-, E(T) = 3/2 RT + 3/2 RT + ZPE + 0 + E_ele
    For H+, E(T) = 0      + 3/2 RT + ZPE + 0 + E_ele
    
Hence, neglecting the contribution due to the vibrations as argued above, 
one obtains the following expression for the proton affinity

    P(A) = -ΔE + RT = -ΔE_ele -ΔZPE + 3/2RT + RT = -ΔE_ele -ΔZPE + 5/2RT

To determine the proton affinity one therefore has to calculate two contributions: 
the change in electronic energy given by (note that the contribution of the proton is zero)
and the difference in zero point energies.

In order to calculate the energies of AH and A- you have to first optimize both systems. 
The ZPE can then be obtained by calculating the normal modes of the system via determination 
of the Hessian matrix.
"""

import psi4

from .geometry import Geometry
from .intermediate import Intermediate


class ProtonAffinity:
    G_sol_proton = -262.4 # (kcal/mol)
    def __init__(self, 
                 geometry: Geometry,
                 functional: str = 'wb97x-d',
                 basis: str = '6-113+G(2d,2p)',
                 scale_factor: float = 1.0,
                 memory: str = '4 GB',
                 num_threads: int = 4,
                ):
        self.geometry = geometry
        self.functional = functional
        self.basis = basis
        self.scale_factor = scale_factor
        self.memory = memory
        self.num_threads = num_threads
        self.PA = None
        
        i = Intermediate(geometry.smiles, thiolate_smiles='SC', verbose=True)
        
        carbanion = Geometry(i.carbanion_smiles, -1, 1)
        carbanion.pre_optimize()
        carbanion.optimize()
        # carbanion.optimize(functional='wb97x-d', basis='6-311+G(2d,2p)')

        self.systems = {
            # product
            'HA': {
                'geometry': self.geometry,
                'E': None,
                'G': None,
            },
            # carbanion intermediate
            'A-': {
                'geometry': carbanion,
                'E': None,
                'G': None,
            }
        }


    def run(self) -> float:
        for state, datadict in self.systems:
            datadict['G'] = datadict['geometry'].gibbs_free_energy(
                functional=self.functional, 
                basis=self.basis,
                scale_factor=self.scale_factor,
                memory=self.memory, 
                num_threads=self.num_threads)

        self.PA = psi4.constants.hartree2kcalmol * (self.systems['HA']['G'] - self.systems['A-']['G'])
        self.PA = self.PA - ProtonAffinity.G_sol_proton
        
        return self.PA