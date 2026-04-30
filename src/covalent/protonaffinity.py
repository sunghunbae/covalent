
import psi4



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
        G_neut = Gibbs_free_energy(mol_neutral, 
                                    scale_factor=0.970, 
                                    functional=OPT_METHOD, 
                                    basis=OPT_BASIS, 
                                    temperature=TEMPERATURE)
        desc.G_neutral = G_neut
    else:
        print("  [1/2] Re-using previously computed G(neutral) …")
        G_neut = desc.G_neutral

    # ── Protonated species ────────────────────────────────────────────────
    print("  [2/2] Optimising protonated species (charge=+1) …")
    start_xyz = protonated_xyz if protonated_xyz else neutral_xyz
    mol_prot = make_psi4_mol(start_xyz, charge=+1, multiplicity=1)
    G_prot = Gibbs_free_energy(mol_prot, 
                                scale_factor=0.970, 
                                functional=OPT_METHOD, 
                                basis=OPT_BASIS, 
                                temperature=TEMPERATURE)
    desc.G_protonated = G_prot

    # PA (proton affinity)
    # Standard definition: PA = –ΔH_deprotonation ≈ –ΔG at β-carbon site
    # Here computed as: ΔG_protonation = G(MH+) – G(M) – G(H+)
    # A more negative value → stronger base / better Michael acceptor
    PA = (G_prot - G_neut) * HARTREE_TO_KCAL - G_PROTON_KCAL
    desc.proton_affinity_kcal = PA
    print(f"  Proton Affinity (β-carbon) = {PA:.2f} kcal/mol")

    return desc
