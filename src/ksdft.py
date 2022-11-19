import psi4
import interface_psi4 as ipsi4


class driver():
  def __init__(self, nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name):
    self._nuclear_numbers = nuclear_numbers
    self._geom_coordinates = geom_coordinates
    self._basis_set_name = basis_set_name
    self._ksdft_functional_name = ksdft_functional_name


  def ks_scf(self):
    # Not direct SCF

    proc_ao_integral = ipsi4.proc_psi4(
        self._nuclear_numbers, self._geom_coordinates,
        self._basis_set_name, self._ksdft_functional_name)

    # Compute all requisite analytical AO integrations
    ao_kinetic_integral = proc_ao_integral.ao_kinetic_integral()
    ao_hartree_potential_integral = proc_ao_integral.ao_hartree_potential_integral()
    ap_electron_repulsion_integral = proc_ao_integral.ap_electron_repulsion_integral()
    ao_overlap_integral = proc_ao_integral.ao_overlap_integral()
