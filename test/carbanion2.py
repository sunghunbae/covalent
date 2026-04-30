"""
Michael Acceptor Carbanion Formation Energy Calculator
using Psi4 for covalent warhead reactivity screening.

Reaction modeled:
  R-CH=CH-EWG  +  RS⁻  →  RS-CH₂-C⁻H-EWG  (carbanion intermediate)

We approximate reactivity via:
  ΔE_carbanion = E(carbanion) - E(neutral acceptor)

A more negative ΔE indicates a more stable carbanion → higher reactivity.
"""

import psi4
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdchem import Mol

# ── Psi4 global settings ──────────────────────────────────────────────────────
psi4.set_memory("4 GB")
psi4.set_num_threads(4)
psi4.set_options({
    "basis": "6-31+G*",          # diffuse functions important for anions
    "scf_type": "df",
    "reference": "uhf",          # UHF for open-shell carbanion (if doublet)
    "guess": "sad",
    "d_convergence": 1e-6,
    "e_convergence": 1e-8,
    "maxiter": 150,
})

THEORY = "b3lyp"                 # DFT with diffuse basis handles anions well
# For higher accuracy: "wb97x-d/6-31+G*" or "mp2/6-31+G*"


# ── Geometry preparation via RDKit ────────────────────────────────────────────

def smiles_to_psi4_geometry(smiles: str, charge: int = 0, mult: int = 1,
                             optimize_mmff: bool = True) -> str:
    """
    Convert SMILES to a Psi4-formatted geometry string.
    Generates 3D coordinates via RDKit MMFF94 optimization.
    
    Args:
        smiles:       SMILES string of the molecule
        charge:       formal charge (0 = neutral, -1 = carbanion)
        mult:         spin multiplicity (1 = singlet, 2 = doublet)
        optimize_mmff: pre-optimize with MMFF94 force field
    
    Returns:
        Psi4 geometry block string
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    
    if optimize_mmff:
        result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        if result != 0:
            print(f"  Warning: MMFF optimization did not fully converge (code={result})")
    
    conf = mol.GetConformer()
    lines = [f"{charge} {mult}"]
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        lines.append(f"  {symbol:2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")
    
    lines.append("units angstrom")
    return "\n".join(lines)


def build_carbanion_smiles(michael_acceptor_smiles: str,
                            beta_carbon_idx: int = None) -> str:
    """
    Build the SMILES for the carbanion intermediate formed after thiolate
    attacks the β-carbon of the Michael acceptor.

    The intermediate is: RS-CHR-C⁻R'-EWG
    We model the carbanion WITHOUT the thiolate (i.e., the β-carbanion fragment
    that results from proton abstraction / charge placement at α-carbon).

    For simplicity in a screening context, we model carbanion formation as:
      neutral molecule → molecule with one extra electron (radical anion)
    OR
      explicitly build α-carbanion SMILES by modifying the structure.

    Here we use the electron-addition model: ΔE = E(anion radical, -1) - E(neutral).
    This avoids the need to specify exact atom indices and is robust for screening.
    
    Returns the same SMILES (charge handled via Psi4 geometry charge parameter).
    """
    # For screening purposes: same geometry, different charge/multiplicity
    # This models vertical electron affinity at the beta carbon region
    return michael_acceptor_smiles


# ── Single-point energy calculation ──────────────────────────────────────────

def compute_energy(smiles: str, charge: int, mult: int,
                   label: str, theory: str = THEORY) -> float:
    """
    Run a Psi4 single-point energy calculation.
    Optionally performs geometry optimization if opt=True.
    
    Returns energy in Hartrees.
    """
    print(f"\n  [{label}] Setting up geometry (charge={charge}, mult={mult})...")
    
    geom_str = smiles_to_psi4_geometry(smiles, charge=charge, mult=mult)
    mol = psi4.geometry(geom_str)
    
    # Output file per calculation to avoid clutter
    psi4.core.set_output_file(f"psi4_{label.replace(' ', '_')}.out", False)
    
    print(f"  [{label}] Running {theory.upper()}/6-31+G* single point...")
    energy = psi4.energy(theory, molecule=mol)
    
    print(f"  [{label}] E = {energy:.8f} Hartrees")
    return energy


def compute_carbanion_formation_energy(
        name: str,
        neutral_smiles: str,
        theory: str = THEORY,
        model: str = "vertical_ea",   # "vertical_ea" or "explicit_carbanion"
        optimize: bool = False,
    ) -> dict:
    """
    Calculate carbanion formation energy for a Michael acceptor.

    Two models are supported:

    1. "vertical_ea"  (recommended for screening):
       ΔE = E(radical anion, charge=-1, mult=2) - E(neutral, charge=0, mult=1)
       This is the vertical electron affinity. More negative = more electrophilic.
       Fast, no structural editing needed.

    2. "explicit_carbanion":
       Models the α-carbon carbanion after thiolate addition:
       ΔE = E(closed-shell carbanion, charge=-1, mult=1) - E(neutral, charge=0, mult=1)
       Requires a pre-built carbanion SMILES where α-carbon bears the negative charge.

    Args:
        name:           human-readable name for the warhead
        neutral_smiles: SMILES of the neutral Michael acceptor
        theory:         DFT functional or wavefunction method string
        model:          "vertical_ea" or "explicit_carbanion"
    
    Returns:
        dict with energies and ΔE in kcal/mol
    """
    print(f"\n{'='*60}")
    print(f"  Warhead: {name}")
    print(f"  SMILES:  {neutral_smiles}")
    print(f"  Model:   {model}")
    print(f"{'='*60}")
    
    # Neutral ground state
    if optimize:
        compute_with_optimization(name, neutral_smiles, theory=theory, model=model)
    else:
        e_neutral = compute_energy(
            neutral_smiles, charge=0, mult=1,
            label=f"{name}_neutral", theory=theory
        )
    
    if model == "vertical_ea":
        # Radical anion (one extra electron, doublet)
        e_anion = compute_energy(
            neutral_smiles, charge=-1, mult=2,
            label=f"{name}_radical_anion", theory=theory
        )
        delta_e_hartree = e_anion - e_neutral
        
    elif model == "explicit_carbanion":
        # Closed-shell carbanion SMILES must be supplied separately
        # Here we use the same geometry; user should supply modified SMILES
        carbanion_smiles = build_carbanion_smiles(neutral_smiles)
        e_carbanion = compute_energy(
            carbanion_smiles, charge=-1, mult=1,
            label=f"{name}_carbanion", theory=theory
        )
        delta_e_hartree = e_carbanion - e_neutral
    else:
        raise ValueError(f"Unknown model: {model}")
    
    HARTREE_TO_KCAL = 627.5094740631
    delta_e_kcal = delta_e_hartree * HARTREE_TO_KCAL
    
    print(f"\n  ΔE_carbanion = {delta_e_kcal:.2f} kcal/mol")
    print(f"  (More negative → more reactive toward thiolate)")
    
    return {
        "name": name,
        "smiles": neutral_smiles,
        "model": model,
        "E_neutral_Ha": e_neutral,
        "E_anion_Ha": e_anion if model == "vertical_ea" else None,
        "E_carbanion_Ha": e_carbanion if model == "explicit_carbanion" else None,
        "delta_E_Ha": delta_e_hartree,
        "delta_E_kcal_mol": delta_e_kcal,
    }


# ── Geometry optimization (optional, for higher accuracy) ────────────────────

def compute_with_optimization(name: str, 
                              smiles: str,
                              theory: str = THEORY,
                              model: str | None = None) -> dict:
    """
    Higher-accuracy workflow:
    1. Optimize neutral geometry at theory/6-31+G*
    2. Optimize carbanion geometry at theory/6-31+G*
    3. Compute ΔE from relaxed geometries (adiabatic EA)
    
    More expensive but accounts for geometric relaxation upon charging.
    """
    print(f"\n  Running geometry optimizations for {name}...")
    
    geom_neutral = smiles_to_psi4_geometry(smiles, charge=0, mult=1)
    mol_neutral = psi4.geometry(geom_neutral)
    psi4.core.set_output_file(f"psi4_{name}_opt_neutral.out", False)
    e_neutral = psi4.optimize(theory, molecule=mol_neutral)
    
    geom_anion = smiles_to_psi4_geometry(smiles, charge=-1, mult=2)
    mol_anion = psi4.geometry(geom_anion)
    psi4.core.set_output_file(f"psi4_{name}_opt_anion.out", False)
    e_anion = psi4.optimize(theory, molecule=mol_anion)
    
    delta_e_kcal = (e_anion - e_neutral) * 627.5094740631
    print(f"  Adiabatic ΔE = {delta_e_kcal:.2f} kcal/mol (relaxed geometries)")
    
    return {
        "name": name,
        "smiles": smiles,
        "model": "adiabatic_ea",
        "delta_E_kcal_mol": delta_e_kcal,
    }


# ── Screening workflow ────────────────────────────────────────────────────────

def screen_warheads(warhead_library: list[dict],
                    theory: str = THEORY,
                    model: str = "vertical_ea") -> list[dict]:
    """
    Screen a library of Michael acceptor warheads by carbanion formation energy.
    
    Args:
        warhead_library: list of {"name": str, "smiles": str}
        theory:          Psi4-compatible method string
        model:           "vertical_ea" or "explicit_carbanion"
    
    Returns:
        Results sorted by ΔE (most reactive first)
    """
    results = []
    for warhead in warhead_library:
        try:
            result = compute_carbanion_formation_energy(
                name=warhead["name"],
                neutral_smiles=warhead["smiles"],
                theory=theory,
                model=model,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {warhead['name']}: {e}")
            results.append({
                "name": warhead["name"],
                "smiles": warhead["smiles"],
                "delta_E_kcal_mol": np.nan,
                "error": str(e),
            })
    
    # Sort by ΔE ascending (most negative = most reactive)
    results.sort(key=lambda x: x.get("delta_E_kcal_mol", np.inf))
    return results


def print_ranking(results: list[dict]):
    """Pretty-print reactivity ranking."""
    print(f"\n{'='*70}")
    print(f"  REACTIVITY RANKING (by carbanion formation energy)")
    print(f"  More negative ΔE → more reactive toward Cys thiolate")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Name':<25} {'ΔE (kcal/mol)':<18} {'SMILES'}")
    print(f"  {'-'*65}")
    for i, r in enumerate(results, 1):
        dE = r.get("delta_E_kcal_mol", float("nan"))
        dE_str = f"{dE:.2f}" if not np.isnan(dE) else "ERROR"
        print(f"  {i:<5} {r['name']:<25} {dE_str:<18} {r['smiles']}")


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Example warhead library (common covalent drug warheads targeting Cys)
    # β-carbon is the electrophilic site in each case
    warhead_library = [
        # Acrylamide — classic mild warhead (e.g., ibrutinib) low to moderate reactivity
        {"name": "acrylamide",          "smiles": "C=CC(=O)N"},
        # N-alkylacrylamide - low to moderate reactivity, used in reversible covalent drugs
        {"name": "n_alkylacrylamide",  "smiles": "C=CC(=O)NC"},
        # 2-Chloroacrylamide - more reactive
        {"name": "chloroacrylamide",    "smiles": "C=C(Cl)C(=O)N"},
        # a-Cyanoacrylamide - highly reactive
        {"name": "a_cyanoacrylamide",   "smiles": "C=C(C#N)C(=O)N"},
        # a-Cyanoacrylate - highly reactive
        {"name": "a_cyanoacrylate",     "smiles": "C=C(C#N)C(=O)O"},
        # Methacrylamide - low reactivity due to steric hindrance
        {"name": "methacrylamide",      "smiles": "C=C(C)C(=O)N"},
        # Allenamide - highly reacive
        {"name": "allenamide",          "smiles": "C=C=CN"},
        # Vinyl sulfonamide - highly reactive
        {"name": "vinyl_sulfonamide",   "smiles": "C=CS(=O)(=O)N"},
        # Vinyl sulfone — more reactive
        {"name": "vinyl_sulfone",        "smiles": "C=CS(=O)(=O)c1ccccc1"},
        # Cyanoacrylamide — highly reactive, used in reversible-covalent drugs
        {"name": "cyanoacrylamide",      "smiles": "N/C(=C/C#N)C(=O)N"},
        # Propiolamide — alkyne-based warhead
        {"name": "propiolamide",         "smiles": "C#CC(=O)N"},
        # Acrylic acid — reference
        {"name": "acrylic_acid",         "smiles": "C=CC(=O)O"},
        # Vinyl ketone — reactive
        {"name": "methyl_vinyl_ketone",  "smiles": "C=CC(=O)C"},
    ]
    
    # Run screening
    results = screen_warheads(
        warhead_library,
        # theory="b3lyp",      # swap to "wb97x-d" for better anion description
        theory="wb97x-d",      # swap to "wb97x-d" for better anion description
        model="vertical_ea"
    )
    
    # Print ranked results
    print_ranking(results)
    
    # Save to CSV
    import csv
    with open("warhead_reactivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "delta_E_kcal_mol"])
        writer.writeheader()
        writer.writerows(results)
    print("\n  Results saved to warhead_reactivity.csv")