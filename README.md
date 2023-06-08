# SimpleQC
A simple code of electronic structure theories for developing new methodologies and educational purposes.

## Feature
This code can be applied to generic molecules and basis sets thanks to the Psi4's atomic orbital (AO) integral modules. For the simplicity, we tried redundant coding.

In the current version, restricted and unrestricted Hartree-Fock theory calculations for arbitrary spin states and molecular charges can be performed. Ground-state total energy, electronic energy, nuclei-repulsion energy, molecular orbital (MO) energies and coefficients, analytical integrals for Hartree-Fock theory, density matrix (in AO basis), and Mulliken atomic charges are available.

For the post-Hartree-Fock methods, configuration interaction singles (CIS) theory for excited-state calculations and second-order Møller-Plesset perturbation theory (MP2) are also implemented. They can be only applied to spin singlet states.

The Kohn-Sham density functional theory (KS-DFT) calculations can be performed with the local density approximation (LDA) functional for the exchange(-correlation) potential and energy. The correlation part is neglected. Only the closed-shell systems can be calculated. The other functionals are not implemented.

As an example of effective models at quantum chemistry level of theory, the electronic Coulomb interaction energy between two molecules can be calculated based on the Hartree-Fock theory.

## Requirements
Psi4: for AO integral and for generating numerical grids and weights  
Basis_Set_Exchange: for processing molecular parameters and basis sets

## Contributors
Takafumi Shiraogawa (@takafumi-shiraogawa)  
Kai Oshiro (@Kai-Oshiro)

## Inputs
sqc.conf: Configuration file of SimpleQC  
*.xyz: XYZ file of a molecular geometry