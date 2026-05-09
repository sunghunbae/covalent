"""
Michael addition of Cys-SH (Cys-S-) to an α,β-unsaturated warhead proceeds 
via nucleophilic attack at the b-carbon (conjugate addition). Since finding 
the transition state conformation is computationally extensive iterative process,
we want to calculates QM descriptors that may correlate with the activation energy
for cysteine-targeting electrophilic warheads.

It has been reported that the formation energy of α-carbanion intermediate energy 
correlates with the TS barrier.

Michael acceptor  +  MeS-  -> Carbanion intermediate (anion at α-carbon) -> Me-S-adduct

Carbanion Formation Free Energy (ΔG) = ΔG(Carbanion) - ΔG(Michael_acceptor) - ΔG(MeS-)

Faster Screening Proxy
Carbanion Single-Point Energy (ΔE) = ΔE(Carbanion) - ΔE(Michael_acceptor) - ΔE(MeS-)
"""

import psi4

from .geometry import Geometry
from .reaction import Reaction
from pathlib import Path


def thiolate_formation():
    # thiol = Geometry("[H]SC([H])([H])[H]")
    # thiol.xtb_optimize()
    # thiol.optimize()
    # E_thiol = thiol.single_point_energy()

    thiolate = Geometry("[H]C([H])([H])[S-]")
    thiolate.xtb_optimize()
    thiolate.optimize()
    
    E_thiolate = thiolate.single_point_energy()
    


def carbanion_formation(smiles: str) -> float:

    rxn = Reaction(smiles, thiol_smiles="SC")
    assert rxn.reactant_smiles
    assert rxn.carbanion_smiles
    assert rxn.carbanion_charge == -1

    reactant = Geometry(rxn.reactant_smiles)
    reactant.xtb_optimize()
    reactant.optimize()

    carbanion = Geometry(rxn.carbanion_smiles, charge=-1)
    carbanion.xtb_optimize()
    carbanion.optimize()

    E_reactant = reactant.single_point_energy(
        functional="wb97x-d",
        basis="6-311+G(2d,2p)",
        solvent="water",
        solvation_model="pcm",
        memory= "8 GB",
        num_threads= 8)
    
    E_carbanion = carbanion.single_point_energy(
        functional="wb97x-d",
        basis="6-311+G(2d,2p)",
        solvent="water",
        solvation_model="pcm",
        memory= "8 GB",
        num_threads= 8)