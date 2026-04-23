import psi4
from covalent import Geometry


def electrophilicity_index(geometry: Geometry, method: str = 'b3lyp', basis: str = '6-31G*', solvent: str = None):
    """
    Compute Global Electrophilicity Index following BI 2019/2020 protocol.
    
    BI papers used: B3LYP/6-31G* (gas phase) or PCM(MeOH) for solvent effects.
    """

    psi4.set_memory('4 GB')
    psi4.set_num_threads(4)
    psi4.set_output_file('psi4_output.dat', False)

    # Optional: implicit solvent (CPCM/PCM)
    if solvent:
        psi4.set_options({
            'pcm': True,
            'pcm_scf_type': 'total',
        })
        psi4.pcm_helper(f"""
            Units = Angstrom
            Medium {{
              SolverType = CPCM
              Solvent = {solvent}
            }}
            Cavity {{
              RadiiSet = UFF
              Type = GePol
              Scaling = False
              Area = 0.3
            }}
        """)

    psi4.set_options({
        'basis': basis,
        'scf_type': 'df',  # density fitting = faster
        'dft_spherical_points': 302,
        'dft_radial_points': 75,
    })

    # Run geometry optimization first
    psi4.set_options({'geom_maxiter': 200})
    energy, wfn = psi4.optimize(f'{method}/{basis}', molecule=geometry.psi4_mol, return_wfn=True)

    # Extract orbital energies (in Hartree)
    epsilon = wfn.epsilon_a()          # alpha orbital energies
    nmo = wfn.nmo()
    nalpha = wfn.nalpha()

    homo_energy = epsilon.get(nalpha - 1)   # HOMO (0-indexed)
    lumo_energy = epsilon.get(nalpha)        # LUMO

    # Convert to eV
    hartree_to_ev = 27.2114
    e_homo = homo_energy * hartree_to_ev
    e_lumo = lumo_energy * hartree_to_ev

    # GEI formula: ω = (E_HOMO + E_LUMO)^2 / (8*(E_LUMO - E_HOMO))
    mu    = (e_homo + e_lumo) / 2.0        # chemical potential
    eta   = (e_lumo - e_homo) / 2.0        # chemical hardness
    omega = (mu ** 2) / (2.0 * eta)        # GEI in eV

    # Ionization potential and electron affinity (Koopmans)
    IP = -e_homo
    EA = -e_lumo

    return {
        'total_energy_hartree': energy,
        'E_HOMO_eV':   e_homo,
        'E_LUMO_eV':   e_lumo,
        'IP_eV':       IP,
        'EA_eV':       EA,
        'mu_eV':       mu,
        'eta_eV':      eta,
        'omega_eV':    omega,
    }