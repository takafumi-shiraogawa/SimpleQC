from .fmat_proc import fgener_cis_hamiltonian_matrix

def gener_cis_hamiltonian_matrix(num_occupied_mo, num_virtual_mo,
                                 orbital_energy_contributions,
                                 mo_integral_iajb, mo_integral_ijab,
                                 cis_hamiltonian):
  return fgener_cis_hamiltonian_matrix(num_occupied_mo, num_virtual_mo,
                                 orbital_energy_contributions,
                                 mo_integral_iajb, mo_integral_ijab,
                                 cis_hamiltonian)