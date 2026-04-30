import numpy as np
import psi4

from psi4.driver.qcdb import vib as qcdb_vib


def Gibbs_free_energy(mol: psi4.core.Molecule,
                    scale_factor: float,
                    functional: str = 'scf',
                    basis: str = 'cc-pVDZ',
                    temperature: float = 298.15,
                    pressure: float = 101325.0,
                    ) -> float:
    """
    Perform a frequency calculation, scale the frequencies by a given factor, and compute thermochemical properties.
    Parameters
    ----------
    mol : psi4.core.Molecule
        The molecule for which to calculate frequencies.
    scale_factor : float
        The factor by which to scale the frequencies.
    functional : str, optional
        The functional to use for the calculation. Default is 'scf'.
    basis : str, optional
        The basis set to use for the calculation. Default is 'cc-pVDZ'.
    temperature : float, optional
        The temperature at which to compute thermochemical properties. Default is 298.15 K.
    pressure : float, optional
        The pressure at which to compute thermochemical properties. Default is 101325.0 Pa.
    Returns
    -------
    float
        The corrected Gibbs free energy.
    """

    psi4.set_options({'basis': basis})

    # 1. Run the frequency calculation
    E_freq, wfn_freq = psi4.frequency(
        f'{functional}/{basis}',
        molecule=mol, 
        return_wfn=True)

    # Step 2: Scale the Hessian by SCALE_FACTOR^2
    H_orig    = np.array(wfn_freq.hessian())          # (3N, 3N) non-mass-weighted, Eh/a0^2
    H_scaled  = H_orig * (scale_factor ** 2)

    # Step 3: Re-run harmonic_analysis with the scaled Hessian
    #         This keeps ALL vibinfo fields (omega, ZPE, force constants) consistent
    mol_psi  = wfn_freq.molecule()
    geom     = np.array(mol_psi.geometry())
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

    # Step 5: Call thermo() with the fully consistent scaled vibinfo
    thermo, thermo_text = qcdb_vib.thermo(
        vibinfo        = scaled_vibinfo,
        T              = temperature,
        P              = pressure,
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
    S       = (TS / temperature) * 627.509 * 4184   # J/mol/K

    # print(thermo_text)
    # print(f"E_elec      = {E_freq:.6f}  Hartree")
    # print(f"E_zpe       = {E_zpe:.6f}  Hartree")
    # print(f"H({temperature} K) = {H_total:.6f}  Hartree")
    # print(f"G({temperature} K) = {G_total:.6f}  Hartree")
    # print(f"TS          = {TS * 627.509:.4f}  kcal/mol")

    return G_total