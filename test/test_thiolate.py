from covalent import Geometry
from pathlib import Path
import tempfile
import psi4
import os

def thiolate_formation():

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['PSI_SCRATCH'] = temp_dir

        # thiol = Geometry("[H]SC([H])([H])[H]", charge=0)
        # thiol.xtb_optimize()
        # thiol.optimize(functional='wb97x-d', basis='6-311+G(d,p)')

        # E_thiol = thiol.single_point_energy(functional='wb97x-d',
        #                                 basis='6-311+G(d,p)',
        #                                 solvent='water',
        #                                 num_threads=10)
    

        thiolate = Geometry("[H]C([H])([H])[S-]", charge=-1)
        thiolate.xtb_optimize()
        thiolate.optimize(functional='wb97x-d', basis='6-311+G(d,p)')

        E_thiolate = thiolate.single_point_energy(functional='wb97x-d', 
                                              basis='6-311+G(d,p)',
                                              solvent='water',
                                              num_threads=10)
    
        # print("E_thiol=", E_thiol)
        print("E_thiolate=", E_thiolate)
        # print("ΔE (kcal/mol)=", psi4.constants.hartree2kcalmol * (E_thiolate - E_thiol))


if __name__ == '__main__':
    thiolate_formation()