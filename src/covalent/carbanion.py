"""
=============================================================================
Covalent Warhead Reactivity Prediction via Psi4
=============================================================================
Calculates quantum chemical descriptors correlated with Michael addition
reaction energy barriers for cysteine-targeting electrophilic warheads:

  1. Carbanion Formation Free Energy (ΔG_carbanion)   — anion at β-carbon
  2. Carbanion Formation Single-Point Energy (ΔE_SP)  — fast screening proxy
  3. Proton Affinity (PA)                              — β-carbon basicity

Theory:
  - Michael addition of Cys-SH to an α,β-unsaturated warhead proceeds via
    nucleophilic attack at the β-carbon (conjugate addition).
  - The β-carbanion intermediate energy correlates with the TS barrier
    (Hammond postulate / BEP relationship).
  - Proton affinity of the β-carbon also correlates because both reflect
    the intrinsic electrophilicity / LUMO character at that site.

Reference workflow per warhead molecule (neutral form):
  neutral  →  geometry optimization  →  E(neutral),  G(neutral)
  anion    →  geometry optimization  →  E(anion),    G(anion)
  proton   →  geometry optimization  →  E(protonated), G(protonated)

  ΔG_carbanion  = G(anion)      - G(neutral)          [+ ZPE / thermal]
  ΔE_SP         = E_SP(anion)   - E_SP(neutral)        [single-point, faster]
  PA            = G(protonated) - G(neutral) - G(H+)   [proton affinity]

  G(H+) at 298 K = -6.28 kcal/mol  (translational only, Sackur-Tetrode)

Usage:
  python covalent_warhead_reactivity_psi4.py

Requirements:
  pip install psi4          # or conda install psi4 -c psi4
  pip install rdkit
=============================================================================
"""

import psi4
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os, json, textwrap

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Global Psi4 settings
# ─────────────────────────────────────────────────────────────────────────────

psi4.set_memory("4 GB")
psi4.set_num_threads(20)

# Output file — set to None to print to stdout
PSI4_OUTPUT = "psi4_warhead.out"
if PSI4_OUTPUT:
    psi4.core.set_output_file(PSI4_OUTPUT, False)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Calculation settings  (edit here to change level of theory)
# ─────────────────────────────────────────────────────────────────────────────

# --- Geometry optimisation ---
OPT_METHOD   = "b3lyp"          # DFT functional
OPT_BASIS    = "6-31+G(d)"      # Diffuse functions (+) essential for anions
OPT_DISPERSION = "d3bj"        # Grimme D3(BJ) dispersion correction

# --- Single-point energy (used for ΔE_SP) ---
SP_METHOD    = "wb97x-d"        # Range-separated hybrid with dispersion
SP_BASIS     = "6-311+G(2d,2p)"

# --- Thermochemistry temperature ---
TEMPERATURE  = 298.15           # K

# Proton free energy correction at 298 K (kcal/mol)
# From statistical mechanics: G(H+) = H(H+) - T·S(H+)
# H(H+) = 5/2 RT = 1.48 kcal/mol; S(H+) = 26.05 cal/(mol·K)
G_PROTON_KCAL = -6.28           # standard state 1 atm

HARTREE_TO_KCAL = 627.509474

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper: build Psi4 geometry string
# ─────────────────────────────────────────────────────────────────────────────

def make_psi4_mol(xyz_block: str, charge: int, multiplicity: int) -> psi4.core.Molecule:
    """
    Construct a Psi4 Molecule from an XYZ-format coordinate block.

    Parameters
    ----------
    xyz_block : str
        Atomic coordinates in XYZ format (element  x  y  z, one atom per line).
        Do NOT include the line-count header or comment line — just coordinates.
    charge : int
        Total molecular charge (0 = neutral, -1 = anion, +1 = cation).
    multiplicity : int
        Spin multiplicity (1 = singlet, 2 = doublet, …).

    Returns
    -------
    psi4.core.Molecule
    """
    mol_str = f"{charge} {multiplicity}\n{xyz_block.strip()}"
    return psi4.geometry(mol_str)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core computation functions
# ─────────────────────────────────────────────────────────────────────────────

# Psi4 calculates harmonic vibrational frequencies using the frequency() (or frequencies(), freq()) command, 
# which computes the Hessian matrix (second derivatives of energy with respect to nuclear coordinates) 
# and performs thermochemical analysis. 
# The command is a wrapper for hessian(), usually run after a geometry optimization 
# to ensure the structure is at a local minimum. 


def is_saddle_point(wfn: psi4.core.Wavefunction) -> bool:
    """
    Check if the optimized geometry is a saddle point (TS) by counting imaginary frequencies.

    Returns
    -------
    bool
        True if exactly one imaginary frequency is present, indicating a TS.
    """
    freqs = wfn.frequencies().to_array()
    num_imaginary = np.sum(freqs < 0)
    return num_imaginary == 1


def is_local_minimum(wfn: psi4.core.Wavefunction) -> bool:
    """
    Check if the optimized geometry is a minimum by ensuring no imaginary frequencies.

    Returns
    -------
    bool
        True if all frequencies are real (≥ 0), indicating a minimum.
    """
    freqs = wfn.frequencies().to_array()
    num_imaginary = np.sum(freqs < 0)
    return num_imaginary == 0


def optimize_geometry(mol: psi4.core.Molecule,
                      method: str = OPT_METHOD,
                      basis: str  = OPT_BASIS,
                      dispersion: str = OPT_DISPERSION
                      ) -> tuple[float, float, float]:
    """
    Run geometry optimisation + frequency analysis to obtain thermochemistry.

    Returns
    -------
    (E_electronic_hartree, G_total_hartree, ZPE_hartree)
        E  — SCF / DFT electronic energy at the optimised geometry
        G  — Gibbs free energy (electronic + ZPE + thermal + entropy)
        ZPE — zero-point vibrational energy
    """
    psi4.set_options({
        "basis": basis,
        "dft_dispersion_parameters": [dispersion],   # activates D3(BJ)
        "geom_maxiter": 200,
        "g_convergence": "gau_tight",
    })

    # Optimise then compute frequencies
    E_opt, wfn = psi4.optimize(f"{method}/{basis}", molecule=mol, return_wfn=True)

    # Frequency analysis, reusing gradient from optimization
    # It is slow, but we need it to get ZPE and thermal corrections for ΔG.
    # In Psi4, frequency analysis is essential to convert a single-point electronic energy 
    # into a useful thermodynamic energy at finite temperatures (K). 
    # It calculates vibrational frequencies to determine the 
    # [Zero-Point Vibrational Energy (ZPVE)] and thermal contributions to enthalpy and Gibbs free energy.

    E_freq, wfn_freq = psi4.frequency(
        f"{method}/{basis}", 
        molecule=mol,
        return_wfn=True,
        ref_gradient=wfn.gradient(),
    )

    # Scale frequencies (only positive/real ones)
    SCALE_FACTOR = 0.970
    raw_freqs = wfn_freq.frequencies().to_array()
    scaled_freqs = np.where(raw_freqs > 0, raw_freqs * SCALE_FACTOR, raw_freqs)

    # Apply scaled frequencies or fall back to scaling the Hessian
    try:
        wfn_freq.set_frequencies(psi4.core.Vector.from_array(scaled_freqs))
        psi4.vibanal_wfn(wfn_freq, temperature=TEMPERATURE)
    except AttributeError:
        H_orig = np.array(wfn_freq.hessian())
        H_scaled_psi = psi4.core.Matrix.from_array(H_orig * (SCALE_FACTOR ** 2))
        wfn_freq.set_hessian(H_scaled_psi)
        psi4.vibanal_wfn(wfn_freq, temperature=TEMPERATURE)

    # Extract thermochemical quantities
    thermo  = wfn_freq.frequency_analysis
    ZPE     = thermo["ZPE_vib"].data
    H_corr  = thermo["Hcorr"].data
    G_corr  = thermo["Gcorr"].data

    E_zpe   = E_freq + ZPE
    H_total = E_freq + H_corr
    G_total = E_freq + G_corr
    TS      = H_total - G_total
    S       = (TS / TEMPERATURE) * 627.509 * 4184  # J/mol/K

    print(f"E_elec  = {E_freq:.6f}  Hartree")
    print(f"E_zpe   = {E_zpe:.6f}  Hartree")
    print(f"H({TEMPERATURE}K) = {H_total:.6f}  Hartree")
    print(f"G({TEMPERATURE}K) = {G_total:.6f}  Hartree")
    print(f"TS      = {TS * HARTREE_TO_KCAL:.4f}  kcal/mol")

    return E_freq, G_total, ZPE

    # # Run frequency analysis, passing temperature directly as a keyword arg
    # E_freq, wfn_freq = psi4.frequency(f"{method}/{basis}", molecule=mol, return_wfn=True, temperature=TEMPERATURE)

    # # Psi4 does not natively scale frequencies before computing thermochemistry, 
    # # so if you need scaled ZPE/enthalpy, you'd need to recompute thermochemical 
    # # quantities manually using the scaled frequencies

    # # Apply freq scale factor manually after the fact
    # SCALE_FACTOR = 0.970 # B3LYP/6-31G(d) scale factor
    # raw_freqs = wfn_freq.frequencies().to_array()  # in cm^-1
    # # Only scale real (positive) frequencies
    # scaled_freqs = np.where(raw_freqs > 0, raw_freqs * SCALE_FACTOR, raw_freqs)

    # # Patch scaled frequencies back into the wavefunction
    # scaled_freqs_psi = psi4.core.Vector.from_array(scaled_freqs)
    # try:
    #     wfn_freq.set_frequencies(scaled_freqs_psi)
    # except AttributeError:
    #     print("Warning: Could not set scaled frequencies in Psi4 wavefunction. Thermochemistry may be uncorrected.")
    #     # In this case, the thermochemistry will be computed from the original (unscaled) frequencies.
    #     # Re-run vibrational/thermochemical analysis using the scaled Hessian
    #     # Scale the Hessian by SCALE_FACTOR^2 (since freq ∝ sqrt(k), scaling freq by s = scaling k by s^2)
    #     H_orig = np.array(wfn_freq.hessian())                    # get original Hessian as numpy array
    #     H_scaled = H_orig * (SCALE_FACTOR ** 2)                  # scale the force constants

    #     # Convert back to psi4.Matrix and set it in the wavefunction
    #     H_scaled_psi = psi4.core.Matrix.from_array(H_scaled)
    #     wfn_freq.set_hessian(H_scaled_psi)

    # # Recompute thermochemistry with scaled frequencies
    # psi4.vibanal_wfn(wfn_freq, temperature=TEMPERATURE)

    # # Now extract corrected thermochemical quantities
    # thermo = wfn_freq.frequency_analysis

    # ZPE    = thermo["ZPE_vib"].data   # Zero-point energy correction (Hartree)
    # H_corr = thermo["Hcorr"].data     # Enthalpy correction H - E_elec (Hartree)
    # G_corr = thermo["Gcorr"].data     # Gibbs correction G - E_elec (Hartree)

    # # thermo = wfn_freq.frequency_analysis
    # # ZPE    = thermo["ZPE_corr"].get()            # Hartree
    # # H_corr = thermo["Hcorr"].get()               # Hartree  (H - E_elec)
    # # G_corr = thermo["Gcorr"].get()               # Hartree  (G - E_elec)

    # E_elec  = E_freq                      # Pure electronic energy
    # E_zpe   = E_freq + ZPE                # Electronic + zero-point energy
    # H_total = E_freq + H_corr             # Total enthalpy at temperature T
    # G_total = E_freq + G_corr             # Total Gibbs free energy at temperature T

    # # Entropy contribution (T*S = H - G)
    # TS = H_total - G_total                # Hartree
    # T  = TEMPERATURE                      # K
    # S  = (TS / T) * 627.509 * 4184        # J/mol/K  (optional)

    # print(f"E_elec  = {E_elec:.6f}  Hartree")
    # print(f"E_zpe   = {E_zpe:.6f}  Hartree")
    # print(f"H({T}K) = {H_total:.6f}  Hartree")
    # print(f"G({T}K) = {G_total:.6f}  Hartree")
    # print(f"TS      = {TS*HARTREE_TO_KCAL:.4f}  kcal/mol")

    # G_total = E_freq + G_corr

    # return E_freq, G_total, ZPE


def single_point_energy(mol: psi4.core.Molecule,
                        method: str = SP_METHOD,
                        basis: str  = SP_BASIS) -> float:
    """
    Compute a single-point energy at a previously optimised geometry.

    Returns
    -------
    float : electronic energy in Hartree
    """
    psi4.set_options({"basis": basis})
    E_sp, _ = psi4.energy(f"{method}/{basis}", molecule=mol, return_wfn=True)
    return E_sp


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Descriptor calculations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReactivityDescriptors:
    """Container for all computed reactivity descriptors."""
    name: str
    # Carbanion descriptors
    dG_carbanion_kcal:   Optional[float] = None   # Free energy of anion formation
    dE_SP_carbanion_kcal: Optional[float] = None  # Single-point energy difference
    # Proton affinity
    proton_affinity_kcal: Optional[float] = None  # PA = G(protonated) - G(neutral) - G(H+)
    # Raw energies (Hartree)
    G_neutral:    Optional[float] = None
    G_anion:      Optional[float] = None
    G_protonated: Optional[float] = None
    E_SP_neutral: Optional[float] = None
    E_SP_anion:   Optional[float] = None
    # Qualitative reactivity prediction
    predicted_reactivity: str = "not computed"
    notes: list = field(default_factory=list)


def compute_carbanion_descriptors(
        neutral_xyz: str,
        anion_xyz:   Optional[str] = None,
        name: str = "warhead"
) -> ReactivityDescriptors:
    """
    Calculate carbanion formation free energy and single-point energy difference.

    Parameters
    ----------
    neutral_xyz : str
        XYZ coordinates of the neutral warhead molecule.
    anion_xyz : str, optional
        Initial guess XYZ for the β-carbanion (charge=-1).
        If None, the neutral geometry is used as starting point (may be fine
        for small geometry changes, but a proper anion guess is recommended).
    name : str
        Label for the warhead.

    Returns
    -------
    ReactivityDescriptors with ΔG and ΔE_SP filled in.
    """
    desc = ReactivityDescriptors(name=name)

    print(f"\n{'─'*60}")
    print(f"  Carbanion descriptors: {name}")
    print(f"{'─'*60}")

    # ── Neutral molecule ──────────────────────────────────────────────────
    print("  [1/4] Optimising neutral molecule …")
    mol_neutral = make_psi4_mol(neutral_xyz, charge=0, multiplicity=1)
    E_neut, G_neut, _ = optimize_geometry(mol_neutral)
    desc.G_neutral = G_neut

    # ── Anion (β-carbanion, charge = -1) ─────────────────────────────────
    print("  [2/4] Optimising β-carbanion (charge=-1) …")
    start_xyz = anion_xyz if anion_xyz else neutral_xyz
    mol_anion = make_psi4_mol(start_xyz, charge=-1, multiplicity=1)
    E_anion, G_anion, _ = optimize_geometry(mol_anion)
    desc.G_anion = G_anion

    # ── ΔG_carbanion ─────────────────────────────────────────────────────
    dG = (G_anion - G_neut) * HARTREE_TO_KCAL
    desc.dG_carbanion_kcal = dG
    print(f"  ΔG_carbanion = {dG:.2f} kcal/mol")

    # ── Single-point energies on optimised geometries ────────────────────
    print("  [3/4] Single-point SP energy on neutral geometry …")
    # Re-read optimised Cartesians from the Psi4 molecule objects
    # (they were updated in-place by psi4.optimize)
    E_SP_neut = single_point_energy(mol_neutral)
    desc.E_SP_neutral = E_SP_neut

    print("  [4/4] Single-point SP energy on anion geometry …")
    E_SP_anion = single_point_energy(mol_anion)
    desc.E_SP_anion = E_SP_anion

    dE_SP = (E_SP_anion - E_SP_neut) * HARTREE_TO_KCAL
    desc.dE_SP_carbanion_kcal = dE_SP
    print(f"  ΔE_SP_carbanion = {dE_SP:.2f} kcal/mol")

    return desc


def compute_proton_affinity(
        neutral_xyz:    str,
        protonated_xyz: Optional[str] = None,
        desc: Optional[ReactivityDescriptors] = None,
        name: str = "warhead"
) -> ReactivityDescriptors:
    """
    Calculate the proton affinity (PA) of the β-carbon.

    PA = G(protonated) - G(neutral) - G(H+)

    NOTE: The protonated species represents H+ addition to the β-carbon,
    converting it from sp2 (alkene) to sp3, which mimics the reverse of
    deprotonation.  A more negative PA ↔ higher β-carbon basicity ↔
    greater susceptibility to nucleophilic addition (Michael acceptor).

    Parameters
    ----------
    neutral_xyz : str
        XYZ of the neutral molecule (can be pre-optimised coords).
    protonated_xyz : str, optional
        Initial XYZ for the β-carbon protonated form (charge=+1 for
        overall neutral → +1; for anion → neutral).
        Convention here: neutral warhead → add H+ → cation (+1).
    desc : ReactivityDescriptors, optional
        Existing descriptor object to update (avoids re-optimising neutral).
    """
    if desc is None:
        desc = ReactivityDescriptors(name=name)

    print(f"\n{'─'*60}")
    print(f"  Proton Affinity: {name}")
    print(f"{'─'*60}")

    # ── Neutral (reuse if already computed) ──────────────────────────────
    if desc.G_neutral is None:
        print("  [1/2] Optimising neutral molecule …")
        mol_neutral = make_psi4_mol(neutral_xyz, charge=0, multiplicity=1)
        _, G_neut, _ = optimize_geometry(mol_neutral)
        desc.G_neutral = G_neut
    else:
        print("  [1/2] Re-using previously computed G(neutral) …")
        G_neut = desc.G_neutral

    # ── Protonated species ────────────────────────────────────────────────
    print("  [2/2] Optimising protonated species (charge=+1) …")
    start_xyz = protonated_xyz if protonated_xyz else neutral_xyz
    mol_prot = make_psi4_mol(start_xyz, charge=+1, multiplicity=1)
    _, G_prot, _ = optimize_geometry(mol_prot)
    desc.G_protonated = G_prot

    # PA (proton affinity)
    # Standard definition: PA = –ΔH_deprotonation ≈ –ΔG at β-carbon site
    # Here computed as: ΔG_protonation = G(MH+) – G(M) – G(H+)
    # A more negative value → stronger base / better Michael acceptor
    PA = (G_prot - G_neut) * HARTREE_TO_KCAL - G_PROTON_KCAL
    desc.proton_affinity_kcal = PA
    print(f"  Proton Affinity (β-carbon) = {PA:.2f} kcal/mol")

    return desc


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Reactivity classification (empirical thresholds)
# ─────────────────────────────────────────────────────────────────────────────
# These thresholds are APPROXIMATE and should be calibrated against
# experimental IC50 / kinact data for your specific warhead series.
# Values are informed by B3LYP/6-31+G(d) benchmark studies.

def classify_reactivity(desc: ReactivityDescriptors) -> ReactivityDescriptors:
    """
    Assign a qualitative reactivity label based on computed descriptors.

    Thresholds (B3LYP/6-31+G(d) level, illustrative):
      ΔG_carbanion < 40 kcal/mol  → highly reactive
      40-60 kcal/mol              → moderately reactive
      > 60 kcal/mol               → weakly reactive

    Proton Affinity:
      PA > 200 kcal/mol → strong base at β-C → reactive Michael acceptor
      PA 180-200        → moderate
      PA < 180          → weak
    """
    scores = []

    if desc.dG_carbanion_kcal is not None:
        dG = desc.dG_carbanion_kcal
        if dG < 40:
            scores.append(3)   # high
        elif dG < 60:
            scores.append(2)   # medium
        else:
            scores.append(1)   # low

    if desc.proton_affinity_kcal is not None:
        pa = desc.proton_affinity_kcal
        if pa > 200:
            scores.append(3)
        elif pa > 180:
            scores.append(2)
        else:
            scores.append(1)

    if not scores:
        return desc

    avg = np.mean(scores)
    if avg >= 2.5:
        desc.predicted_reactivity = "HIGH — potent Michael acceptor; monitor selectivity"
    elif avg >= 1.5:
        desc.predicted_reactivity = "MODERATE — typical covalent warhead reactivity"
    else:
        desc.predicted_reactivity = "LOW — may require activation or scaffold modification"

    return desc


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[ReactivityDescriptors]) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print("  COVALENT WARHEAD REACTIVITY REPORT")
    print(f"  Level of theory: {OPT_METHOD}/{OPT_BASIS} // {SP_METHOD}/{SP_BASIS}")
    print(f"  Temperature: {TEMPERATURE} K")
    print(sep)

    header = f"{'Warhead':<22} {'ΔG_carb':>10} {'ΔE_SP':>10} {'PA':>10}  Reactivity"
    print(header)
    print("─" * 70)
    for d in results:
        dG  = f"{d.dG_carbanion_kcal:>10.2f}"   if d.dG_carbanion_kcal   is not None else f"{'N/A':>10}"
        dE  = f"{d.dE_SP_carbanion_kcal:>10.2f}" if d.dE_SP_carbanion_kcal is not None else f"{'N/A':>10}"
        pa  = f"{d.proton_affinity_kcal:>10.2f}" if d.proton_affinity_kcal is not None else f"{'N/A':>10}"
        rx  = d.predicted_reactivity
        print(f"{d.name:<22}{dG}{dE}{pa}  {rx}")

    print("─" * 70)
    print("  Units: kcal/mol")
    print("  ΔG_carb  = Gibbs free energy of β-carbanion formation")
    print("  ΔE_SP    = Single-point energy of β-carbanion formation")
    print("  PA       = Proton affinity at β-carbon")
    print(f"{sep}\n")

    # JSON dump
    out = [
        {
            "name":                   d.name,
            "dG_carbanion_kcal":      d.dG_carbanion_kcal,
            "dE_SP_carbanion_kcal":   d.dE_SP_carbanion_kcal,
            "proton_affinity_kcal":   d.proton_affinity_kcal,
            "predicted_reactivity":   d.predicted_reactivity,
            "raw_G_neutral_hartree":  d.G_neutral,
            "raw_G_anion_hartree":    d.G_anion,
            "raw_G_protonated_hartree": d.G_protonated,
            "raw_E_SP_neutral_hartree": d.E_SP_neutral,
            "raw_E_SP_anion_hartree":   d.E_SP_anion,
        }
        for d in results
    ]
    with open("warhead_reactivity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results written to warhead_reactivity_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Example warheads  (replace with your molecules)
# ─────────────────────────────────────────────────────────────────────────────
# Coordinates below are illustrative (from quick MM pre-optimisation).
# In practice: use RDKit + MMFF94 or xtb to generate starting geometries,
# then pass them here.  The β-carbon is the electrophilic site attacked by
# Cys-SH (conjugated carbon β to the carbonyl/EWG).

# ── Acrylamide  (CH2=CH-C(=O)-NH2) ──────────────────────────────────────────
ACRYLAMIDE_XYZ = """
C   0.000000   0.000000   0.000000
C   1.330000   0.000000   0.000000
C   2.100000   1.220000   0.000000
O   1.560000   2.300000   0.000000
N   3.430000   1.200000   0.000000
H  -0.540000   0.940000   0.000000
H  -0.540000  -0.940000   0.000000
H   1.870000  -0.940000   0.000000
H   3.970000   2.050000   0.000000
H   3.970000   0.350000   0.000000
"""

# ── Acrylamide β-carbanion  (H removed, charge=-1) ───────────────────────────
# β-carbon is C1 (index 1, 1-based) — the one bonded to the carbonyl carbon.
# For initial anion guess we simply use the same geometry; Psi4 will relax it.
ACRYLAMIDE_ANION_XYZ = ACRYLAMIDE_XYZ   # starting guess only

# ── Acrylamide β-protonated (add H+ to β-carbon, charge=+1) ──────────────────
# This becomes CH3-CH2-C(=O)-NH2+ (protonated enamine tautomer).
# Provide a proper geometry from RDKit in a real study.
ACRYLAMIDE_PROTONATED_XYZ = """
C   0.000000   0.000000   0.000000
C   1.540000   0.000000   0.000000
C   2.310000   1.220000   0.000000
O   1.770000   2.300000   0.000000
N   3.640000   1.200000   0.000000
H  -0.390000   1.020000   0.000000
H  -0.390000  -0.510000   0.880000
H  -0.390000  -0.510000  -0.880000
H   1.930000  -0.510000   0.880000
H   1.930000  -0.510000  -0.880000
H   4.180000   2.050000   0.000000
H   4.180000   0.350000   0.000000
"""

# ── Vinyl sulfone  (CH2=CH-SO2-Ph, simplified as CH2=CH-SO2-CH3) ─────────────
VINYL_SULFONE_XYZ = """
C   0.000000   0.000000   0.000000
C   1.330000   0.000000   0.000000
S   2.200000   1.400000   0.000000
O   1.500000   2.500000   0.600000
O   3.600000   1.300000   0.600000
C   2.500000   1.700000  -1.600000
H  -0.540000   0.940000   0.000000
H  -0.540000  -0.940000   0.000000
H   1.870000  -0.940000   0.000000
H   2.000000   2.600000  -1.900000
H   3.580000   1.750000  -1.800000
H   2.200000   0.900000  -2.300000
"""

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from covalent import Geometry

    all_results: list[ReactivityDescriptors] = []

    # ── Warhead 1: Acrylamide ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  WARHEAD 1: Acrylamide")
    print("="*60)

    acrylamide = Geometry(smiles="C=CC(=O)N")
    acrylamide.optimize()


    acrylamide_carbanion = Geometry(smiles="[CH2-]CC(=O)N", charge=-1)
    acrylamide_carbanion.optimize()

    acrylamide_protonated = Geometry(smiles="C=CC(=O)N[H+]", charge=+1)
    acrylamide_protonated.optimize()

    desc_acrylamide = compute_carbanion_descriptors(
        neutral_xyz = acrylamide.xyz_block,
        anion_xyz   = acrylamide_carbanion.xyz_block,
        name        = "Acrylamide"
    )
    desc_acrylamide = compute_proton_affinity(
        neutral_xyz    = acrylamide.xyz_block,
        protonated_xyz = acrylamide_protonated.xyz_block,
        desc           = desc_acrylamide,
        name           = "Acrylamide"
    )
    desc_acrylamide = classify_reactivity(desc_acrylamide)
    all_results.append(desc_acrylamide)

    # ── Warhead 2: Vinyl Sulfone ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  WARHEAD 2: Vinyl Sulfone")
    print("="*60)

    desc_vs = compute_carbanion_descriptors(
        neutral_xyz = VINYL_SULFONE_XYZ,
        anion_xyz   = VINYL_SULFONE_XYZ,   # use neutral as anion starting guess
        name        = "Vinyl Sulfone"
    )
    # Proton affinity for vinyl sulfone (protonated XYZ not shown; use neutral)
    desc_vs = compute_proton_affinity(
        neutral_xyz    = VINYL_SULFONE_XYZ,
        protonated_xyz = VINYL_SULFONE_XYZ,
        desc           = desc_vs,
        name           = "Vinyl Sulfone"
    )
    desc_vs = classify_reactivity(desc_vs)
    all_results.append(desc_vs)

    # ── Final Report ──────────────────────────────────────────────────────
    print_report(all_results)
