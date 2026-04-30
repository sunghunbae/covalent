import psi4
import copy
import numpy as np
from qcelemental import Datum
from psi4.driver.qcdb import vib as qcdb_vib

# 1. Define molecule and run frequency calculation
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")
psi4.set_options({'basis': 'cc-pVDZ'})

# 1. Run the frequency calculation
E_freq, wfn_freq = psi4.frequency(
    'scf/cc-pvdz',
    molecule=mol, 
    return_wfn=True)

# 3. Create scaled frequency data
scaling_factor = 0.96
# Step 2: Scale the Hessian by SCALE_FACTOR^2
H_orig    = np.array(wfn_freq.hessian())          # (3N, 3N) non-mass-weighted, Eh/a0^2
H_scaled  = H_orig * (scaling_factor ** 2)

# Step 3: Re-run harmonic_analysis with the scaled Hessian
#         This keeps ALL vibinfo fields (omega, ZPE, force constants) consistent
mol_psi  = wfn_freq.molecule()
geom     = np.array(mol_psi.geometry())
import sys; sys.exit()  # STOP HERE FOR TESTING

mass     = np.array([mol_psi.mass(i) for i in range(mol_psi.natom())])  # in u
basisset = wfn_freq.basisset()
irrep_labels = wfn_freq.molecule().irrep_labels()

scaled_vibinfo, _ = qcdb_vib.harmonic_analysis(
    hess         = H_scaled,
    geom         = geom,
    mass         = mass,
    basisset     = basisset,
    irrep_labels = irrep_labels,
)

# Step 4: Extract molecular properties for thermo()
molecular_mass = sum(mol_psi.mass(i) for i in range(mol_psi.natom()))
multiplicity   = mol_psi.multiplicity()
sigma          = mol_psi.rotational_symmetry_number()
rot_const       = np.asarray(mol_psi.rotational_constants())

TEMPERATURE = 298.15  # K
PRESSURE    = 101325.0  # Pa
# Step 5: Call thermo() with the fully consistent scaled vibinfo
thermo, thermo_text = qcdb_vib.thermo(
    vibinfo        = scaled_vibinfo,
    T              = 298.15,
    P              = 101325.0,
    multiplicity   = multiplicity,
    molecular_mass = molecular_mass,
    E0             = E_freq,
    sigma          = sigma,
    rot_const      = rot_const,
)

# The keys in thermo are:
# ['E0', 'B', 'sigma', 'T', 'P', 
#   'S_elec', 'S_trans', 'Cv_trans', 'Cp_trans', 'E_trans', 'H_trans', 
#   'S_rot', 'Cv_rot', 'Cp_rot', 'E_rot', 'H_rot', 
#   'S_vib', 'Cv_vib', 'Cp_vib', 'ZPE_vib', 'E_vib', 'H_vib', 
#   'H_elec', 'G_elec', 'G_trans', 'G_rot', 'G_vib', 'Cv_elec', 'Cp_elec', 
#   'ZPE_elec', 'E_elec', 'ZPE_trans', 'ZPE_rot', 
#   'S_tot', 'Cv_tot', 'Cp_tot', 'ZPE_corr', 'ZPE_tot', 
#   'E_corr', 'E_tot', 'H_corr', 'H_tot', 'G_corr', 'G_tot'])

# Step 6: Extract results
ZPE    = thermo["ZPE_vib"].data
H_corr = thermo["H_corr"].data
G_corr = thermo["G_corr"].data

E_zpe   = E_freq + ZPE
H_total = E_freq + H_corr
G_total = E_freq + G_corr
TS      = H_total - G_total
S       = (TS / TEMPERATURE) * 627.509 * 4184   # J/mol/K

print(thermo_text)
print(f"E_elec      = {E_freq:.6f}  Hartree")
print(f"E_zpe       = {E_zpe:.6f}  Hartree")
print(f"H({TEMPERATURE}K) = {H_total:.6f}  Hartree")
print(f"G({TEMPERATURE}K) = {G_total:.6f}  Hartree")
print(f"TS          = {TS * 627.509:.4f}  kcal/mol")