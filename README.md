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