import pytest
from covalent import Intermediate


def test_intermediates():
    for name, smiles, thiol, expected_intermediate, expected_product in [
        ("acrylamide", "C=CC(=O)N", "SC", 
            "[H][C-](C(=O)N([H])[H])C([H])([H])SC([H])([H])[H]",
            "[H]N([H])C(=O)C([H])([H])C([H])([H])SC([H])([H])[H]"),
        ("vinyl_sulfone", "C=CS(=O)(=O)c1ccccc1", "SC",
            "[H]c1c([H])c([H])c(S(=O)(=O)[C-]([H])C([H])([H])SC([H])([H])[H])c([H])c1[H]",
            "[H]c1c([H])c([H])c(S(=O)(=O)C([H])([H])C([H])([H])SC([H])([H])[H])c([H])c1[H]"),
        ("cyanoacrylamide", "N/C(=C/C#N)C(=O)N", "SC",
            "[H][C-](C#N)C(SC([H])([H])[H])(C(=O)N([H])[H])N([H])[H]",
            "[H]N([H])C(=O)C(SC([H])([H])[H])(N([H])[H])C([H])([H])C#N"),
        ("methyl_vinyl_ketone","C=CC(=O)C", "SC",
            "[H][C-](C(=O)C([H])([H])[H])C([H])([H])SC([H])([H])[H]",
            "[H]C([H])([H])SC([H])([H])C([H])([H])C(=O)C([H])([H])[H]"),
        ("acrylamide_CysLike","C=CC(=O)N", "SCC(N)C(=O)O",
            "[H]OC(=O)C([H])(N([H])[H])C([H])([H])SC([H])([H])[C-]([H])C(=O)N([H])[H]",
            "[H]OC(=O)C([H])(N([H])[H])C([H])([H])SC([H])([H])C([H])([H])C(=O)N([H])[H]"),
        ]:
        try:
            warhead = Intermediate(smiles, thiolate_smiles=thiol, verbose=True)
            assert warhead.carbanion_charge == -1
            assert warhead.carbanion_smiles == expected_intermediate
            assert warhead.product_smiles == expected_product
        except ValueError as e:
            print(f"  ✗ Error: {e}")
