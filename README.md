# SimpleQC
A simple code of electronic structure theories for developing new methodologies and educational purposes.

## Feature
This code can be applied to generic molecules and basis sets thanks to the Psi4's atomic orbital (AO) integral modules. In the current version, restricted and unrestricted Hartree-Fock theory calculations for arbitrary spin states and molecular charges can be performed. Ground-state total energy, electronic energy, nuclei-repulsion energy, molecular orbital (MO) energies and coefficients, analytical AO and MO integrals for Hartree-Fock theory, one-particle reduced density matrix (in AO basis), and Mulliken atomic charges are available. For the post-Hartree-Fock methods, configurational interaction singles (CIS) theory for excited-state calculations and second-order Møller-Plesset perturbation theory (MP2) are also implemented. They can be only applied to spin singlet states.

## Requirements
Psi4: for AO integral, numerical quadrature, and processing exchange-correlation potential  
Basis_Set_Exchange: for processing molecular parameters and basis sets

## Contributors
Takafumi Shiraogawa (@takafumi-shiraogawa)  
Kai Oshiro (@Kai-Oshiro)


## Inputs
sqc.conf: Configuration file of SimpleQC  
*.xyz: XYZ file of a molecular geometry