from .prune import prune
from .reaction import Reaction
from .michaeladdition import MichaelAddition
from .geometry import Geometry
from .fukui import FukuiIndex
from .electrophilicity import omega
from .protonaffinity import ProtonAffinity
from .thermodynamics import Gibbs_free_energy

import psi4

hartree2kcalmol = psi4.constants.hartree2kcalmol

# electronic energies calculated with wb97x-d / 6-311+G(d,p)
# solvent='water', solvent_model='pcm'

E_methyl_thiol = -438.71047767702316 # hatree
E_methyl_thiolate = -438.21834768481375 # hatree
G_methyl_thiol = -438.68873613008606 # hatree
G_methyl_thiolate = -437.98571016382937 # hatree

# Thiol E energy= -438.71047767702316 hartree -275294.981 kcal/mol
# Thiolate E energy= -438.21834768481375 hartree -274986.165 kcal/mol
# E_thiol - E_thiolate= -0.49212999220941356 hartree -308.816 kcal/mol

# Thiol G energy= -438.68873613008606 hartree -275281.338 kcal/mol
# Thiolate G energy= -437.98571016382937 hartree -274840.183 kcal/mol
# G_thiol - G_thiolate= -0.7030259662566891 hartree -441.155 kcal/mol