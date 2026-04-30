import os
import resource
import subprocess
import json
import tempfile
import logging
import shutil
import re
import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D


logger = logging.getLogger(__name__)


# In ASE, the default energy unit is eV (electron volt).
# It will be converted to kcal/mol
# CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026


pt = Chem.GetPeriodicTable()


class GFN2xTB:
    def __init__(
        self,
        molecule: Chem.Mol | None = None,
        ncores: int | None = None,
        xtb_exec: str | Path | None = None,
    ):
        """_summary_

        Args:
            molecule (Chem.Mol): input molecule in rdkit.Chem.Mol format
            ncores (int | None, optional): number of cores for parallelization. Defaults to None (all).
            xtb_exec (str | Path | None, optional): path to xtb executable. Defaults to None.
        """
        if molecule is not None:
            assert isinstance(molecule, Chem.Mol), (
                "molecule must be rdkit.Chem.Mol type"
            )
            assert molecule.GetConformer().Is3D(), "molecule must be a 3D conformer"

            self.rdmol = molecule
            self.charge = rdmolops.GetFormalCharge(self.rdmol)
            self.natoms = molecule.GetNumAtoms()
            self.symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            self.numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
            self.positions = molecule.GetConformer().GetPositions().tolist()

        if ncores is None:
            ncores = os.cpu_count()

        # Parallelisation
        # https://xtb-docs.readthedocs.io/en/latest/setup.html#parallelisation
        os.environ["OMP_STACKSIZE"] = "4G"
        os.environ["OMP_NUM_THREADS"] = f"{ncores},1"
        # OMP_NUM_THREADS=4,2 would mean the outermost parallel region uses 4 threads,
        # and any parallel regions nested within it would use up to 2 threads.
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        os.environ["MKL_NUM_THREADS"] = f"{ncores}"

        # unlimit the system stack
        resource.setrlimit(
            resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )

        self.xtb_exec = None
        self.cpx_exec = None

        if xtb_exec is None:
            self.xtb_exec = shutil.which("xtb")
            self.cpx_exec = shutil.which("cpx")
        else:
            self.xtb_exec = Path(xtb_exec).resolve().as_posix()
            self.cpx_exec = Path(xtb_exec).resolve().parent.joinpath("cpx").as_posix()

    def is_xtb_ready(self) -> bool:
        """Check if xtb is available.

        Returns:
            bool: True if `xtb` is available, False otherwise.
        """
        return self.xtb_exec is not None

    def is_optimize_ready(self) -> bool:
        try:
            h2o = [
                "$coord",
                " 0.00000000000000      0.00000000000000     -0.73578586109551      o",
                " 1.44183152868459      0.00000000000000      0.36789293054775      h",
                "-1.44183152868459      0.00000000000000      0.36789293054775      h",
                "$end",
            ]

            with tempfile.TemporaryDirectory() as temp_dir:
                test_geometry = os.path.join(temp_dir, "coord")
                with open(test_geometry, "w") as f:
                    f.write("\n".join(h2o))
                proc = subprocess.run(
                    [self.xtb_exec, test_geometry, "--opt"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                assert proc.returncode == 0

            return True

        except:
            print(
                """                          
Conda installed xTB has the Fortran runtime error in geometry optimization. 
Please install xtb using the compiled binary:
                    
$ wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
$ tar -xf xtb-6.7.1-linux-x86_64.tar.xz
$ cp -r xtb-dist/bin/*      /usr/local/bin/
$ cp -r xtb-dist/lib/*      /usr/local/lib/
$ cp -r xtb-dist/include/*  /usr/local/include/
$ cp -r xtb-dist/share      /usr/local/ """
            )

            return False

    def is_cpx_ready(self) -> bool:
        """Checks if the CPCM-X command-line tool, `cpx`, is accessible in the system.

        Returns:
            bool: True if the cpx is found, False otherwise.
        """
        return self.cpx_exec is not None

    def is_cpcmx_ready(self) -> bool:
        """Checks if xtb works with the `--cpcmx` option.

        xtb distributed by the conda does not include CPCM-X function (as of June 17, 2025).
        xtb installed from the github source codes by using meson and ninja includes it.

        Returns:
            bool: True if the --cpcmx option is working, False otherwise.
        """
        if self.is_xtb_ready():
            with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
                cmd = [self.xtb_exec, "--cpcmx"]
                proc = subprocess.run(
                    cmd, cwd=temp_dir, capture_output=True, text=True, encoding="utf-8"
                )
                # we are expecting an error because no input file is given
                assert proc.returncode != 0
                for line in proc.stdout.split("\n"):
                    line = line.strip()
                    if "CPCM-X library was not included" in line:
                        return False

        return True

    def is_ready(self) -> bool:
        """Check if `xtb` and `cpx` are accessible and `xtb --cpcmx` are available.

        Returns:
            bool: True if both `xtb` and `cpx` are accessible, False otherwise.
        """
        return all(
            [
                self.is_xtb_ready(),
                self.is_cpx_ready(),
                self.is_cpcmx_ready(),
                self.is_optimize_ready(),
            ]
        )

    def version(self) -> str | None:
        """Check xtb version.

        Returns:
            str | None: version statement.
        """
        if self.is_xtb_ready():
            with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
                cmd = [self.xtb_exec, "--version"]
                proc = subprocess.run(
                    cmd, cwd=temp_dir, capture_output=True, text=True, encoding="utf-8"
                )
                assert proc.returncode == 0, "GFN2xTB() Error: xtb not available"
                match = re.search("xtb\s+version\s+(?P<version>[\d.]+)", proc.stdout)
                if match:
                    return match.group("version")

        return None

    def to_xyz(self) -> str:
        """Export to XYZ formatted string.

        Returns:
            str: XYZ formatted string
        """
        lines = [f"{self.natoms}", " "]
        for e, (x, y, z) in zip(self.symbols, self.positions):
            lines.append(f"{e:5} {x:23.14f} {y:23.14f} {z:23.14f}")

        return "\n".join(lines)

    def to_turbomole_coord(self, bohr: bool = False) -> str:
        """Returns TURBOMOLE coord file formatted strings.

        Turbomole coord file format:

            - It starts with the keyword `$coord`.
            - Each line after the $coord line specifies an atom, consisting of:
                - Three real numbers representing the Cartesian coordinates (x, y, z).
                - A string for the element name.
                - Optional: an "f" label at the end to indicate that the atom's coordinates are frozen during optimization.
            - Coordinates can be given in Bohr (default), Ångström (`$coord angs`), or fractional coordinates (`$coord frac`).
            - Optional data groups like periodicity (`$periodic`), lattice parameters (`$lattice`), and cell parameters (`$cell`) can also be included.
            - Regarding precision:
                The precision of the coordinates is crucial for accurate calculations, especially geometry optimizations.
                Tools like the TURBOMOLEOptimizer might check for differences in atomic positions with a tolerance of 1e-13.

        Args:
            bohr (bool): whether to use Bohr units of the coordinates. Defaults to False.
                Otherwise, Angstrom units will be used.

        Returns:
            str: TURBOMOLE coord formatted file.
        """
        if bohr:
            lines = ["$coord"]
        else:
            lines = ["$coord angs"]

        for (x, y, z), e in zip(self.positions, self.symbols):
            lines.append(f"{x:20.15f} {y:20.15f} {z:20.15f} {e}")

        lines.append("$end")

        return "\n".join(lines)

    def load_xyz(self, geometry_input_path: Path) -> Chem.Mol:
        """Load geometry.

        Args:
            geometry_input_path (Path): pathlib.Path to the xyz

        Returns:
            Chem.Mol: rdkit Chem.Mol object.
        """
        rdmol_opt = Chem.Mol(self.rdmol)
        with open(geometry_input_path, "r") as f:
            for lineno, line in enumerate(f):
                if lineno == 0:
                    assert int(line.strip()) == self.natoms
                    continue
                elif lineno == 1:  # comment or title
                    continue
                (symbol, x, y, z) = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                atom = rdmol_opt.GetAtomWithIdx(lineno - 2)
                assert symbol == atom.GetSymbol()
                rdmol_opt.GetConformer().SetAtomPosition(
                    atom.GetIdx(), Point3D(x, y, z)
                )

        return rdmol_opt

    def load_wbo(self, wbo_path: Path) -> dict[tuple[int, int], float]:
        """Load Wiberg bond order.

        singlepoint() creates a wbo output file.

        Args:
            wbo_path (Path): path to the wbo file.

        Returns:
            dict(tuple[int, int], float): { (i, j) : wbo, ... } where i and j are atom indices for a bond.
        """

        with open(wbo_path, "r") as f:
            # Wiberg bond order (WBO)
            Wiberg_bond_orders = {}
            for line in f:
                line = line.strip()
                if line:
                    # wbo output has 1-based indices
                    (i, j, wbo) = line.split()
                    # changes to 0-based indices
                    i = int(i) - 1
                    j = int(j) - 1
                    # wbo ouput indices are ascending order
                    ij = (i, j) if i < j else (j, i)
                    Wiberg_bond_orders[ij] = float(wbo)

            return Wiberg_bond_orders

    def singlepoint(
        self, water: str | None = None, verbose: bool = False
    ) -> SimpleNamespace:
        """Calculate single point energy.

        Total energy from xtb output in atomic units (Eh, hartree) is converted to kcal/mol.

        Options:
            ```sh
            -c, --chrg INT
                specify molecular charge as INT, overrides .CHRG file and xcontrol option

            --scc, --sp
                performs a single point calculation

            --gfn INT
                specify parametrisation of GFN-xTB (default = 2)

            --json
                write xtbout.json file

            --alpb SOLVENT [STATE]
                analytical linearized Poisson-Boltzmann (ALPB) model,
                available solvents are acetone, acetonitrile, aniline, benzaldehyde,
                benzene, ch2cl2, chcl3, cs2, dioxane, dmf, dmso, ether, ethylacetate, furane,
                hexandecane, hexane, methanol, nitromethane, octanol, woctanol, phenol, toluene,
                thf, water.
                The solvent input is not case-sensitive. The Gsolv
                reference state can be chosen as reference, bar1M, or gsolv (default).

            -g, --gbsa SOLVENT [STATE]
                generalized born (GB) model with solvent accessable surface (SASA) model,
                available solvents are acetone, acetonitrile, benzene (only GFN1-xTB), CH2Cl2,
                CHCl3, CS2, DMF (only GFN2-xTB), DMSO, ether, H2O, methanol,
                n-hexane (only GFN2-xTB), THF and toluene.
                The solvent input is not case-sensitive.
                The Gsolv reference state can be chosen as reference, bar1M, or gsolv (default).

            --cosmo SOLVENT/EPSILON
                domain decomposition conductor-like screening model (ddCOSMO),
                available solvents are all solvents that are available for alpb.
                Additionally, the dielectric constant can be set manually or an ideal conductor
                can be chosen by setting epsilon to infinity.

            --tmcosmo SOLVENT/EPSILON
                same as --cosmo, but uses TM convention for writing the .cosmo files.

            --cpcmx SOLVENT
                extended conduction-like polarizable continuum solvation model (CPCM-X),
                available solvents are all solvents included in the Minnesota Solvation Database.
            ```

        Args:
            water (str, optional) : water solvation model (choose 'gbsa' or 'alpb')
                alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
                gbsa: generalized Born (GB) model with Surface Area contributions.

        Returns:
            SimpleNamespace(PE(total energy in kcal/mol), charges, wbo)
        """

        with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
            workdir = Path(temp_dir)

            geometry_input_path = workdir / "geometry.xyz"
            xtbout_path = workdir / "xtbout.json"
            wbo_path = workdir / "wbo"
            geometry_output_path = workdir / "xtbtopo.mol"

            with open(geometry_input_path, "w") as geometry:
                geometry.write(self.to_xyz())

            cmd = [self.xtb_exec, geometry_input_path.as_posix()]
            options = ["-c", str(self.charge), "--sp", "--gfn", "2", "--json"]

            if water is not None and isinstance(water, str):
                if water == "gbsa":
                    options += ["--gbsa", "H2O"]
                    # it does not provide Gsolv contribution to the total energy
                elif water == "alpb":
                    options += ["--alpb", "water"]
                    # it does not provide Gsolv contribution to the total energy
                elif water == "cpcmx" and self.is_cpcmx_ready():
                    options += ["--cpcmx", "water"]

            if verbose:
                logger.info(f"singlepoint() {' '.join(cmd + options)}")

            # 'xtbout.json', 'xtbrestart', 'xtbtopo.mol', 'charges', and 'wbo' files will be
            # created in the current working directory.
            proc = subprocess.run(
                cmd + options,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            # if proc.returncode == 0:
            #     print("Standard Output:")
            #     print(proc.stdout)
            # else:
            #     print("Error:")
            #     print(proc.stderr)

            if proc.returncode == 0:
                if xtbout_path.is_file():
                    with open(xtbout_path, "r") as f:
                        datadict = json.load(f)  # takes the file object as input

                Gsolv = None

                if water is not None:
                    #  Free Energy contributions:                       [Eh]        [kcal/mol]
                    # -------------------------------------------------------------------------
                    #  solvation free energy (dG_solv):             -0.92587E-03    -0.58099
                    #  gas phase energy (E)                         -0.52068E+01
                    # -------------------------------------------------------------------------
                    #  total free energy (dG)                       -0.52077E+01
                    for line in proc.stdout.splitlines():
                        if "solvation free energy" in line:
                            m = re.search(
                                r"solvation free energy \(dG_solv\)\:\s+[-+]?\d*\.?\d+E[-+]?\d*\s+(?P<kcalpermol>[-+]?\d*\.?\d+)",
                                line,
                            )
                            Gsolv = float(m.group("kcalpermol"))

                Wiberg_bond_orders = self.load_wbo(wbo_path)

                return SimpleNamespace(
                    natoms=self.natoms,
                    charge=self.charge,
                    PE=datadict["total energy"] * hartree2kcalpermol,
                    Gsolv=Gsolv,
                    charges=datadict["partial charges"],
                    wbo=Wiberg_bond_orders,
                )

        # something went wrong if it reaches here
        return SimpleNamespace()

    def optimize(
        self, water: str | None = None, verbose: bool = False
    ) -> SimpleNamespace:
        """Optimize geometry.

        Options:
            ```sh
            -c, --chrg INT
              specify molecular charge as INT, overrides .CHRG file and xcontrol option
            -o, --opt [LEVEL]
              call ancopt(3) to perform a geometry optimization, levels from crude, sloppy,
              loose, normal (default), tight, verytight to extreme can be chosen
            --gfn INT
              specify parametrisation of GFN-xTB (default = 2)
            --json
              write xtbout.json file
            ```

        Notes:
            Conda installed xtb has Fortran runtime error when optimizing geometry.
            ```sh
            Fortran runtime errror:
                At line 852 of file ../src/optimizer.f90 (unit = 6, file = 'stdout')
                Fortran runtime error: Missing comma between descriptors
                (1x,"("f7.2"%)")
                            ^
                Error termination.
            ```
            
            Default `xtb molecule.xyz --opt` command-line behavior:
                Default parameters (from xtb docs, --opt normal):
                Optimizer       : ANCopt (Approximate Normal Coordinate RFO)
                Method          : GFN2-xTB
                Opt level       : normal
                Energy conv     : 5e-6 Eh
                Gradient conv   : 1e-3 Eh/Bohr
                SCC accuracy    : 1.0
                SCC max iter    : 250
                Electronic temp : 300 K (xtb default for production; 5000 K used internally for convergence aid)
                ANC microcycles : 20
                Max RF displ    : 1.0 Bohr
                Max opt cycles  : auto (min 200, max 10000, based on degrees of freedom)

        Args:
            water (str, optional) : water solvation model (choose 'gbsa' or 'alpb')
                alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
                gbsa: generalized Born (GB) model with Surface Area contributions.

        Returns:
            (total energy in kcal/mol, optimized geometry)
        """
        with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
            workdir = Path(temp_dir)

            geometry_input_path = workdir / "geometry.xyz"
            xtbout_path = workdir / "xtbout.json"
            geometry_output_path = workdir / "xtbopt.xyz"
            wbo_path = workdir / "wbo"

            with open(geometry_input_path, "w") as geometry:
                geometry.write(self.to_xyz())

            cmd = [self.xtb_exec, geometry_input_path.as_posix()]
            options = ["-c", str(self.charge), "-o", "normal", "--gfn", "2", "--json"]

            if water is not None and isinstance(water, str):
                if water == "gbsa":
                    options += ["--gbsa", "H2O"]
                elif water == "alpb":
                    options += ["--alpb", "water"]
                elif water == "cpcmx":
                    logger.warning(
                        "CPCM-X not implemented for geometry optimization. "
                        "Please use another solvation model for optimization instead."
                    )

            if verbose:
                logger.info(f"optimize() {' '.join(cmd + options)}")

            proc = subprocess.run(
                cmd + options,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if proc.returncode == 0 and xtbout_path.is_file():
                with open(xtbout_path, "r") as f:
                    datadict = json.load(f)  # takes the file object as input

                Wiberg_bond_orders = self.load_wbo(wbo_path)
                rdmol_opt = self.load_xyz(geometry_output_path)

                return SimpleNamespace(
                    natoms=self.natoms,
                    charge=self.charge,
                    PE=datadict["total energy"] * hartree2kcalpermol,
                    charges=datadict["partial charges"],
                    wbo=Wiberg_bond_orders,
                    geometry=rdmol_opt,
                )

        # something went wrong if it reaches here
        return SimpleNamespace()

    @staticmethod
    def read_esp_data(filename: str | Path) -> np.ndarray:
        """
        Read xtb ESP surface file with format: x y z esp_value
        """
        points = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        esp = float(parts[3])
                        points.append((x, y, z, esp))
                    except ValueError:
                        continue

        return np.array(points)

    def create_regular_grid(
        self, surface_points, grid_spacing=0.6, padding=4.0, verbose: bool = False
    ):
        """
        Create a regular 3D grid encompassing the molecule.

        Args:
            surface_points: Nx4 array of surface points (x, y, z, esp)
            atoms: List of atomic coordinates
            grid_spacing: Spacing between grid points (in Bohr or Angstrom)
            padding: Extra space around molecule
        """
        # Get molecule bounds
        all_coords = np.array(self.positions)

        # Determine grid bounds with padding
        min_coords = np.min(all_coords, axis=0) - padding
        max_coords = np.max(all_coords, axis=0) + padding

        # Also consider surface points
        surface_coords = surface_points[:, :3]
        min_coords = np.minimum(min_coords, np.min(surface_coords, axis=0) - padding)
        max_coords = np.maximum(max_coords, np.max(surface_coords, axis=0) + padding)

        # Create grid
        x_range = np.arange(min_coords[0], max_coords[0] + grid_spacing, grid_spacing)
        y_range = np.arange(min_coords[1], max_coords[1] + grid_spacing, grid_spacing)
        z_range = np.arange(min_coords[2], max_coords[2] + grid_spacing, grid_spacing)

        nx, ny, nz = len(x_range), len(y_range), len(z_range)

        if verbose:
            logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {nx * ny * nz} points")
            logger.info(f"Grid spacing: {grid_spacing:.3f}")
            logger.info(
                f"Grid bounds: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], "
                f"Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], "
                f"Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]"
            )

        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        grid_points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

        origin = [x_range[0], y_range[0], z_range[0]]
        grid_vectors = [
            [grid_spacing, 0.0, 0.0],
            [0.0, grid_spacing, 0.0],
            [0.0, 0.0, grid_spacing],
        ]

        return grid_points, (nx, ny, nz), origin, grid_vectors

    @staticmethod
    def interpolate_esp_to_grid(
        surface_points, grid_points, method="nearest", max_distance=2.0
    ):
        """
        Interpolate surface ESP values to regular grid using nearest neighbor.
        Points far from surface are set to zero.

        Args:
            surface_points: Nx4 array (x, y, z, esp)
            grid_points: Mx3 array of grid coordinates
            method: Interpolation method ('nearest', 'linear', or 'cubic')
            max_distance: Maximum distance for interpolation (points further get 0)
        """
        surface_coords = surface_points[:, :3]
        esp_values = surface_points[:, 3]

        logger.info(
            f"Interpolating {len(grid_points)} grid points from {len(surface_points)} surface points..."
        )

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(surface_coords)

        # Find distances to nearest surface point
        distances, indices = tree.query(grid_points, k=1)

        # Interpolate using scipy
        if method == "nearest":
            grid_esp = esp_values[indices]
        else:
            grid_esp = griddata(
                surface_coords, esp_values, grid_points, method=method, fill_value=0.0
            )

        # Zero out points far from surface
        grid_esp[distances > max_distance] = 0.0

        logger.info(
            f"ESP range on grid: {np.min(grid_esp):.4f} to {np.max(grid_esp):.4f}"
        )
        logger.info(
            f"Points within {max_distance:.1f} of surface: {np.sum(distances <= max_distance)}"
        )

        return grid_esp

    def to_cube(self, grid_dims, origin, grid_vectors, values) -> str:
        """Write Gaussian cube file format."""
        nx, ny, nz = grid_dims
        cube = []
        cube.append("Cube file: ESP interpolated from xtb surface data")
        cube.append("Electrostatic potential (a.u.)")

        # Number of atoms and origin
        cube.append(
            f"{self.natoms:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}"
        )

        # Grid dimensions and vectors
        cube.append(
            f"{nx:5d} {grid_vectors[0][0]:12.6f} {grid_vectors[0][1]:12.6f} {grid_vectors[0][2]:12.6f}"
        )
        cube.append(
            f"{ny:5d} {grid_vectors[1][0]:12.6f} {grid_vectors[1][1]:12.6f} {grid_vectors[1][2]:12.6f}"
        )
        cube.append(
            f"{nz:5d} {grid_vectors[2][0]:12.6f} {grid_vectors[2][1]:12.6f} {grid_vectors[2][2]:12.6f}"
        )

        # Atom coordinates
        for atomic_num, (x, y, z) in zip(self.numbers, self.positions):
            cube.append(
                f"{atomic_num:5d} {float(atomic_num):12.6f} {x:12.6f} {y:12.6f} {z:12.6f}"
            )

        # ESP values (6 per line)
        for i in range(0, len(values), 6):
            line_values = values[i : i + 6]
            cube.append(" ".join([f"{v:13.5E}" for v in line_values]))

        return "\n".join(cube)

    def esp_surface_points(
        self, water: str | None = None, verbose: bool = False
    ) -> str:
        """Calculate electrostatic potential surface points data.

        Example:
            def esp_to_rgb(value):
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                cmap = cm.get_cmap("seismic")
                rgb = mcolors.to_hex(cmap(value))
                return rgb

            # --- Create 3Dmol viewer
            view = py3Dmol.view(width=600, height=500)
            view.addModel(xyz_data, "xyz")
            view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
            # --- Add ESP points as colored spheres
            for xi, yi, zi, vi in zip(x, y, z, norm):
                color = esp_to_rgb(vi)
                view.addSphere({
                    'center': {'x': float(xi), 'y': float(yi), 'z': float(zi)},
                    'radius': 0.2,
                    'color': color,
                    'opacity': 0.9
                })
            view.zoomTo()
            view.show()
        """
        esp_surface_points_data = ""
        with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
            workdir = Path(temp_dir)
            if verbose:
                logger.info(f"xtb.optimize workdir= {temp_dir}")

            geometry_input_path = workdir / "geometry.xyz"
            xtb_esp_dat = workdir / "xtb_esp.dat"

            with open(geometry_input_path, "w") as geometry:
                geometry.write(self.to_xyz())

            cmd = [self.xtb_exec, geometry_input_path.as_posix()]

            options = ["--sp", "--esp", "--gfn", "2"]

            if water is not None and isinstance(water, str):
                if water == "gbsa":
                    options += ["--gbsa", "H2O"]
                elif water == "alpb":
                    options += ["--alpb", "water"]

            proc = subprocess.run(
                cmd + options,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            # output files: xtb_esp.cosmo, xtb_esp.dat, xtb_esp_profile.dat

            if proc.returncode == 0 and xtb_esp_dat.is_file():
                with open(xtb_esp_dat, "r") as f:
                    esp_surface_points_data = f.read()

        return esp_surface_points_data

    def esp_volumetric(
        self,
        water: str | None = None,
        max_iterations: int = 250,
        grid_spacing: float = 0.6,
        padding: float = 4.0,
        max_distance: float = 2.0,
        method: str = "nearest",
        verbose: bool = False,
    ) -> str:
        """Calculate electrostatic potential volumetric data.

        Example:
            import py3Dmol

            view = py3Dmol.view(width=600, height=400)
            view.addModel(xyz_data, 'xyz')
            view.setStyle({'stick': {}})
            view.addVolumetricData(esp_voldata, 'cube', {'isoval': esp_scale_max*0.5, 'color': 'blue', 'opacity': 0.75})
            view.addVolumetricData(esp_voldata, 'cube', {'isoval': esp_scale_min*0.5, 'color': 'red',  'opacity': 0.75})
            view.zoomTo()
            view.show()
        """
        with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
            workdir = Path(temp_dir)
            if verbose:
                logger.info(f"xtb.optimize workdir= {temp_dir}")

            geometry_input_path = workdir / "geometry.xyz"  # input

            with open(geometry_input_path, "w") as geometry:
                geometry.write(self.to_xyz())

            cmd = [self.xtb_exec, geometry_input_path.as_posix()]

            options = [
                "--sp",
                "--esp",
                "--gfn",
                "2",
                "--iterations",
                str(max_iterations),
            ]

            if water is not None and isinstance(water, str):
                if water == "gbsa":
                    options += ["--gbsa", "H2O"]
                elif water == "alpb":
                    options += ["--alpb", "water"]
                elif water == "cpcmx" and self.is_cpcmx_ready():
                    options += ["--cpcmx", "water"]

            proc = subprocess.run(
                cmd + options,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            try:
                # output files: xtb_esp.cosmo, xtb_esp.dat, xtb_esp_profile.dat
                xtb_esp_dat = workdir / "xtb_esp.dat"  # expected output
                assert proc.returncode == 0
                assert xtb_esp_dat.is_file()
                surface_points = GFN2xTB.read_esp_data(xtb_esp_dat)
                if verbose:
                    logger.info(f"Loaded {len(surface_points)} surface points")
                    logger.info(
                        f"ESP range: {np.min(surface_points[:, 3]):.4f} to {np.max(surface_points[:, 3]):.4f}"
                    )
                grid_points, grid_dims, origin, grid_vectors = self.create_regular_grid(
                    surface_points, grid_spacing, padding
                )
                if verbose:
                    logger.info(
                        f"Grid: {grid_dims[0]} x {grid_dims[1]} x {grid_dims[2]} points"
                    )
                grid_esp = GFN2xTB.interpolate_esp_to_grid(
                    surface_points, grid_points, method, max_distance
                )
                return self.to_cube(grid_dims, origin, grid_vectors, grid_esp)
            except Exception as e:
                logger.error(f"Failed to generate ESP volumetric data ({e}).")
                return ""
