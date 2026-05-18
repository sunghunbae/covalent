import psi4
import time

from importlib.resources import files
from covalent import Geometry, hartree2kcalmol

psi4.core.set_output_file('test_methyl_thiol.dat', True)
psi4.set_num_threads(20)

start_time = time.perf_counter()

# thiol = Geometry("SC", charge=0, mult=1)
# thiol.xtb_optimize()
# thiol.optimize(solvent='water', num_threads=20, max_iter=300)
# thiol.write_xyz("test_thiol.xyz", True)
xyz_string = files('covalent').joinpath('methylthiol/methyl_thiol_optimized.xyz').read_text()
thiol = Geometry(xyz_string=xyz_string)

# thiolate = Geometry("[S-]C", charge=-1, mult=1)
# thiolate.xtb_optimize()
# thiolate.optimize(solvent='water', num_threads=20, max_iter=300)
# thiolate.write_xyz("test_thiolate.xyz", True)
xyz_string = files('covalent').joinpath('methylthiol/methyl_thiolate_optimized.xyz').read_text()
thiolate = Geometry(xyz_string=xyz_string, charge=-1)

E_thiol = thiol.single_point_energy(solvent='water', num_threads=20) # hartree
E_thiolate = thiolate.single_point_energy(solvent='water', num_threads=20) # hartree
G_thiol = thiol.gibbs_free_energy(num_threads=20) # hartree
G_thiolate = thiolate.gibbs_free_energy(num_threads=20) # hartree 

end_time = time.perf_counter()

print(f"Thiol/thiolate completed {(end_time - start_time):.3f} sec")

print(f"Thiol E energy= {E_thiol} hartree {E_thiol*hartree2kcalmol:.3f} kcal/mol")
print(f"Thiolate E energy= {E_thiolate} hartree {E_thiolate*hartree2kcalmol:.3f} kcal/mol")
print(f"E_thiol - E_thiolate= {E_thiol - E_thiolate} hartree {(E_thiol-E_thiolate)*hartree2kcalmol:.3f} kcal/mol")
print(f"Thiol G energy= {G_thiol} hartree {G_thiol*hartree2kcalmol:.3f} kcal/mol")
print(f"Thiolate G energy= {G_thiolate} hartree {G_thiolate*hartree2kcalmol:.3f} kcal/mol")
print(f"G_thiol - G_thiolate= {G_thiol - G_thiolate} hartree {(G_thiol-G_thiolate)*hartree2kcalmol:.3f} kcal/mol")

"""
Thiol E energy= -438.71047767702316 hartree -275294.981 kcal/mol
Thiolate E energy= -438.21834768481375 hartree -274986.165 kcal/mol
E_thiol - E_thiolate= -0.49212999220941356 hartree -308.816 kcal/mol
Thiol G energy= -438.68873613008606 hartree -275281.338 kcal/mol
Thiolate G energy= -437.98571016382937 hartree -274840.183 kcal/mol
G_thiol - G_thiolate= -0.7030259662566891 hartree -441.155 kcal/mol
"""