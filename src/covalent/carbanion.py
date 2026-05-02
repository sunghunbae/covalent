"""
Calculates quantum chemical descriptors correlated with Michael addition
reaction energy barriers for cysteine-targeting electrophilic warheads:

- Michael addition of Cys-SH (Cys-S-) to an α,β-unsaturated warhead proceeds 
    via nucleophilic attack at the b-carbon (conjugate addition).

- The α-carbanion intermediate energy correlates with the TS barrier
    (Hammond postulate / BEP relationship).

    Michael acceptor  +  MeS-  -> Carbanion intermediate (anion at α-carbon)

    Carbanion Formation Free Energy (ΔG) 
    = ΔG(Carbanion) - ΔG(Michael_acceptor) - ΔG(MeS-)

    Faster Screening Proxy

    Carbanion Single-Point Energy (ΔE)
    = ΔE(Carbanion) - ΔE(Michael_acceptor) - ΔE(MeS-)
"""

import psi4
import numpy as np
from .thermodynamics import Gibbs_free_energy

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
    G_neut = Gibbs_free_energy(mol_neutral, 
                                scale_factor=0.970, 
                                functional=OPT_METHOD, 
                                basis=OPT_BASIS, 
                                temperature=TEMPERATURE)
    desc.G_neutral = G_neut

    # ── Anion (β-carbanion, charge = -1) ─────────────────────────────────
    print("  [2/4] Optimising β-carbanion (charge=-1) …")
    start_xyz = anion_xyz if anion_xyz else neutral_xyz
    mol_anion = make_psi4_mol(start_xyz, charge=-1, multiplicity=1)
    G_anion = Gibbs_free_energy(mol_anion, 
                                scale_factor=0.970, 
                                functional=OPT_METHOD, 
                                basis=OPT_BASIS, 
                                temperature=TEMPERATURE)
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