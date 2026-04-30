import psi4
import numpy as np
from pathlib import Path

from covalent import Geometry

"""
    The Fukui (or frontier) function measures the local reactivity of a molecule. 
    It is defined as the change in electron density at each atom when you add or remove an electron.
    The condensed Fukui function for nucleophilic attack fi+ is defined as
    
        f_{i}^{+} = q_{i}(N+1) - q_{i}(N)
        where q_{i}(N) is the charge of atom i in the neutral state of the molecule, 
        and q_{i}(N + 1) is the charge of atom i in the anionic state of the molecule but with the neutral-state structure. 
        
    We will calculate f_{i}^{+} for the Cβ atom in the olefin.
"""


def get_atomic_populations(xyz_block: str, 
                           charge: int, 
                           multiplicity: int, 
                           functional: str = 'HF',
                           basis: str = '6-31G*'):
    """
    Creates the molecule, runs the calculation, and extracts 
    the gross atomic populations (number of electrons per atom).
    """
    # Set resource limits
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('fukui_output.dat', False)
    psi4.set_options({'basis': basis, 'scf__maxiter': 300})
    # psi4.set_options({'basis': basis, 'scf_type': 'df', 'geom_maxiter': 300, 'd_convergence': 1e-8})

    # Create the specialized geometry string with charge and multiplicity
    mol_str = f"{charge} {multiplicity}\n{xyz_block}"
    mol = psi4.geometry(mol_str)
    
    # CRITICAL FIX: Set UHF reference for open-shell states, RHF for closed-shell
    if multiplicity != 1:
        psi4.set_options({'reference': 'uhf'})
    else:
        psi4.set_options({'reference': 'rhf'})
    
    # Calculate energy and return the wavefunction object
    e, wfn = psi4.energy(functional, molecule=mol, return_wfn=True)
    
    # Trigger execution of one-electron property calculations (Mulliken charges)
    psi4.oeprop(wfn, "MULLIKEN_CHARGES")
    
    try:
        net_charges = np.array(wfn.variable("MULLIKEN CHARGES"))
    except KeyError:
        net_charges = np.array(wfn.variable("MULLIKEN_CHARGES"))
    
    # Extract original atomic numbers from the molecule
    atomic_numbers = np.array([wfn.molecule().Z(i) for i in range(wfn.molecule().natom())])
    
    # Gross Population = Atomic Number - Net Charge
    populations = atomic_numbers - net_charges
    
    return populations, wfn.molecule()

# 1. Define your base geometry (without charge and multiplicity)
# Paste your target molecule's coordinates here!
if False:
    xyz_block = """
    O   0.000000   0.000000   0.119720
    H   0.000000   0.761528  -0.478879
    H   0.000000  -0.761528  -0.478879
    """
else:
    m = Geometry("C=CC(=O)N")
    xyz_filename = "acrylamide.xyz"
    if Path(xyz_filename).exists():
        m.update_coords(xyz_filename)
    else:
        m.pre_optimize() # Pre-optimization with GFN2-xTB to get a reasonable starting geometry for Psi4 DFT optimization.
        m.optimize() # B3LYP/6-31G* optimization to get a good geometry for the Fukui calculation
        m.write_xyz(xyz_filename)
    xyz_block = m.xyz_block

# 2. Run calculations for the three states
print("Calculating Neutral state...")
pop_N, molecule = get_atomic_populations(xyz_block, charge=0, multiplicity=1)

print("Calculating Cation (N-1) state...")
pop_N_minus_1, _ = get_atomic_populations(xyz_block, charge=1, multiplicity=1)

print("Calculating Anion (N+1) state...")
pop_N_plus_1, _ = get_atomic_populations(xyz_block, charge=-1, multiplicity=1)

# 3. Compute Fukui Functions
f_plus = pop_N_plus_1 - pop_N       # f+ = q(N) - q(N+1)
f_minus = pop_N_minus_1 - pop_N     # f- = q(N-1) - q(N)
f_zero = (f_plus + f_minus) / 2.0   # f0 = average

# 4. Display the results beautifully
print("\n" + "="*50)
print(f"{'Atom':<6} {'f+ (Nu attack)':<15} {'f- (El attack)':<15} {'f0 (Radical)':<15}")
print("="*50)

for i in range(molecule.natom()):
    symbol = molecule.symbol(i)
    print(f"{symbol:<2} {i:<3} {f_plus[i]:<15.4f} {f_minus[i]:<15.4f} {f_zero[i]:<15.4f}")
print("="*50)
