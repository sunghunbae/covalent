import psi4

"""
PSI4 contains code to interface to the ddx FORTRAN library developed by A. Mikhalev et. al.. The library provides a linear-scaling implementation of standard continuum solvation models using a domain-decomposition ansatz [Cances:2013:054111] [Stamm:2016:054101]. Currently the conductor-like screening model (COSMO) [Klamt:1993:799] [Lipparini:2014:184108], the polarisable continuum model (PCM) [Tomasi:2005:2999] [Nottoli:2019:6061] and the linearized poisson-boltzmann model (LPB) [Lu:2008:973] [Jha:2023:104105] are supported. No additional licence or configuration is required to use ddx with Psi4

The SCF_TYPE parameter in Psi4 determines the algorithm used to calculate the electron repulsion integrals (ERIs) during the Self-Consistent Field (SCF) procedure, which is the foundational step for both Hartree-Fock and Density Functional Theory (DFT) calculations. It defines how the Fock matrix is constructed at each iteration.

SCF_TYPE options:
- PK (Default): A conventional direct SCF method that is robust but can be slower.
- DF (Density Fitting): Approximates 4-center integrals using 3-center integrals, drastically speeding up calculations, especially for large systems.
- DIRECT: Computes all integrals directly on the fly

"""

# methylthiolate molecule (note -1 charge)
mol = psi4.geometry("""
-1 1
 S   0.000000   0.000000   0.000000
 C   0.000000   0.000000   1.810000
 H   1.020000   0.000000   2.170000
 H  -0.510000  -0.880000   2.170000
 H  -0.510000   0.880000   2.170000
 symmetry c1
""")

pcm_input = """
Units = Angstrom
Medium {
    SolverType = IEFPCM
    Solvent = Water
}
Cavity {
    RadiiSet = UFF
    Type = GePol
}
"""

def test_1():
    psi4.set_options({
        'basis': '6-311++G(d,p)',
        'solvation_model': 'smd',  # Selects the SMD parameter set within ddx
        'solvent': 'water'         # Standard water parameters
        })
    # ddx handles the cavity and non-electrostatic terms automatically
    energy = psi4.energy('b3lyp')
    print(f"Total SMD Solvated Energy: {energy} Hartrees")


def test_2():
    psi4.set_options({
        'basis': '6-311++G(d,p)',
        'pcm': True,              # Enable PCM
        'pcm_scf_type': 'total'   # Default handling
    })
    psi4.set_module_options('pcm', pcm_input)
    energy = psi4.energy('b3lyp')
    print(f"Solvated Energy: {energy} Hartrees")

def test_3():
    psi4.set_options({
    "basis": "sto-3g",
    "scf_type": "pk",
    "ddx": True,
    "ddx_model":     "pcm",
    "ddx_solvent":   "water",
    "ddx_radii_set": "uff",
    })

    scf_e = psi4.energy('SCF')
    print(f"solvated energy: {scf_e} Hartrees")


def test_4():
    # --- Single-point energy (used for ΔE_SP) ---
    SP_FUNCTIONAL    = "wb97x-d"        # Range-separated hybrid with dispersion
    SP_BASIS     = "6-311+G(2d,2p)"
    
    psi4.set_options({
    "basis": SP_BASIS,
    "scf_type": "pk",
    "ddx": True,
    "ddx_model":     "pcm",
    "ddx_solvent":   "water",
    "ddx_radii_set": "uff",
    })

    theory_level = f'{SP_FUNCTIONAL}/{SP_BASIS}'
    scf_e = psi4.energy(theory_level)
    print(f"solvated energy: {scf_e} Hartrees")


def test_5():
    # --- Single-point energy (used for ΔE_SP) ---
    SP_FUNCTIONAL    = "wb97x-d"        # Range-separated hybrid with dispersion
    SP_BASIS     = "6-311+G(2d,2p)"
    
    psi4.set_options({
    "basis": SP_BASIS,
    "scf_type": "pk",
    "ddx": True,
    "ddx_model":     "cosmo",
    "ddx_solvent":   "water",
    "ddx_radii_set": "uff",
    })

    theory_level = f'{SP_FUNCTIONAL}/{SP_BASIS}'
    scf_e = psi4.energy(theory_level)
    print(f"solvated energy: {scf_e} Hartrees")


# test_1()
# test_2()
# test_3()
# test_4()
test_5()