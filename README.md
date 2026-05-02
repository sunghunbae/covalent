# Datasets

## Experimental GSH Reactivity

Experimental datasets of GSH t<sub>1/2</sub> (min), along with corresponding molecular structures (SMILES), compiled from the literature by Danilack <i>et al.</i> (2024), were reformatted as CSV files and stored in the `data/` directory. Each CSV file includes the header:

> `Name`,`SMILES`,`GSH_half_life_min`

* A. D. Danilack, et al., Reactivities of acrylamide warheads toward cysteine targets: a QM/ML approach to covalent inhibitor design. J. Comput. Aided Mol. Des. 38, 21 (2024).
* R. A. Ward, et al., Structure- and reactivity-based development of covalent inhibitors of the activating and gatekeeper mutant forms of the epidermal growth factor receptor (EGFR). J. Med. Chem. 56, 7025–7048 (2013).
* M. E. Flanagan, et al., Chemical and computational methods for the characterization of covalent reactive groups for the prospective design of irreversible inhibitors. J. Med. Chem. 57, 10072–10079 (2014).
* V. J. Cee, et al., Systematic study of the glutathione (GSH) reactivity of N-arylacrylamides: 1. Effects of aryl substitution. J. Med. Chem. 58, 9171–9178 (2015).
* R. Lonsdale, et al., Expanding the armory: Predicting and tuning covalent warhead reactivity. J. Chem. Inf. Model. 57, 3124–3137 (2017).

| Filename | Reference |
| -------- | --------- |
| data/Danilack_et_al_2024_table_S2.csv | Ward, <i>et al.</i> (2013) |
| data/Danilack_et_al_2024_table_S3.csv | Flanagan <i>et al.</i> (2014) |
| data/Danilack_et_al_2024_table_S4.csv | Cee <i>et al.</i> (2015) |
| data/Danilack_et_al_2024_table_S5.csv | Lonsdale <i>et al.</i> (2017) |

## Activation Energy (E<sub>a</sub>)

Liu <i>et al.</i> (2023) performed DFT calculations of 30 olefins and demonstrated a useful correlation between activation energy (E<sub>a</sub>) and some QM descriptors such as carbanion formation energy and proton affinity.  Their original supporting information (`data.xlsx`) was reformatted to CSV file including the corresponding molecular structures (SMILES). CSV header is below:
- `compound`
- `comp_HOMO (eV)`
- `comp_LUMO (eV)`
- `comp_SCF (Hartree)`
- `comp_SP (Hartree)`
- `comp_Ca (a.u.)`
- `comp_Cb (a.u.)`
- `dCb`
- `carb_HOMO (eV)`
- `carb_LUMO (eV)`
- `carb_SCF (Hartree)`
- `carb_SP (Hartree)`
- `dSCF_carb (Hartree)`
- `dSP_carb (Hartree)`
- `carb_Ca (a.u.)`
- `carb_Cb (a.u.)`
- `omega_p (eV)`
- `activation_energy (kcal/mol)`
- `carb_S (a.u.)`
- `carb_SC_dis (Å)`
- `carb_formation_energy (kcal/mol)`
- `omega (eV)`
- `comp_Cb_f+ (a.u.)`
- `prod_formation (kcal/mol)`
- `k_298K (s-1)`
- `carb_protonation (kcal/mol)`
- `ts_SCF (Hartree)`
- `ts_SP (Hartree)`
- `prod_SCF (Hartree)`
- `prod_SP (Hartree)`


* R. Liu, E. A. Vázquez-Montelongo, S. Ma, J. Shen, Quantum descriptors for predicting and understanding the structure-activity relationships of Michael acceptor warheads. J. Chem. Inf. Model. 63, 4912–4923 (2023). 
  > All electronic structure calculations were carried out using the ORCA 5.0.1 software package. The ωB97X-D3(BJ) density functional theory (DFT) method with the 6-311+G(d,p) basis set and the Solvent Model Density (SMD) continuum solvent model was used for both geometry optimization and electronic energy calculations.

# Frequency Analysis

Note: Frequency analysis must be performed at the same level of theory as the geometry optimization to ensure consistency.

## Zero-Point Vibrational Energy (ZPE/ZPVE)

Quantum mechanics dictates that molecules vibrate even at 0 K. Frequency analysis computes these vibrational modes to calculate the ZPVE, which is required for accurate ground-state energies (Total Energy = E_elec + ZPE)

## Thermal Corrections (Enthalpy & Free Energy)

To predict thermodynamics at 298.15 K or higher, the frequencies are used to calculate the contribution of vibrational, rotational, and translational degrees of freedom to the heat capacity, enthalpy, and entropy.

## Verification of Stationary Points

The frequency calculation (Hessian) checks if the optimized geometry is a stable minimum (all real, positive frequencies) or a transition state (exactly one imaginary frequency).

## [Thermo Analysis] Automation

When running frequency() in Psi4, the code automatically performs a thermochemical analysis, utilizing the calculated frequencies to return key values like "ZPVE", "THERMAL ENERGY CORRECTION", and "GIBBS FREE ENERGY CORRECTION".

# Scaling Factor

We need a scaling factor for Psi4 frequency analysis to correct for systematic errors arising from the harmonic approximation, finite basis sets, and incomplete electron correlation. Scaling is crucial when comparing calculated harmonic frequencies to experimental fundamental frequencies (e.g., IR/Raman spectra) or for calculating accurate thermodynamic quantities like zero-point energy (ZPE).

[University of Minnesota Frequency Scale Factors](https://comp.chem.umn.edu/freqscale/210722_Database_of_Freq_Scale_Factors_v5.pdf)

[NIST CCCBDB](https://cccbdb.nist.gov/vibscalejustx.asp)

| method | ZPE scaling factor |
| ------ | ------------------ |
| B3LYP/6-31G(2df,2p) | 0.981 |
| B3LYP/6-31G(d) | 0.977 |

## When to Use Scaling Factors:

## Comparing to Experiment: 
When you need to align computed frequencies with experimental fundamental frequencies, as harmonic calculations typically overestimate these values.

## Calculating Thermodynamics: When you need accurate Zero Point Vibrational Energy (ZPVE) or thermodynamic data (enthalpies, free energies). 
Specific Level of Theory: Scaling is necessary for most common density functional theory (DFT) methods (e.g., B3LYP) and smaller basis sets. 

Key Takeaways for Psi4 Frequencies:
Get Specific Factors: Scaling factors depend directly on the functional and basis set used, so you must select a matching pair from databases such as the  
Opt. Geometry Required: Ensure frequencies are calculated at an optimized geometry to avoid meaningless results. 
Approximate Nature: Scaling factors usually provide two significant digits of accuracy and do not remove all errors, reflecting limitations in the level of theory used.