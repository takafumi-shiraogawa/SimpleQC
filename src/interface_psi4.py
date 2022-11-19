import numpy as np
import psi4
from basis_set_exchange import lut

def gener_psi4_geom(nuclear_numbers, geom_coordinates):
  symbol_nuclear_numbers = []
  for i in range(len(nuclear_numbers)):
    symbol_nuclear_numbers.append(lut.element_sym_from_Z(nuclear_numbers[i]))

  return symbol_nuclear_numbers


class proc_psi4():
  def __init__(self, nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name):
    # This redundant operation will be removed.
    mol_xyz = open('temp_psi4_geom.xyz', mode='r').read()
    mol = psi4.geometry(mol_xyz)

    # symbol_nuclear_numbers = gener_psi4_geom(nuclear_numbers, geom_coordinates)
    # mol_list = []
    # for i in range(len(nuclear_numbers)):
    #   mol_list.append([symbol_nuclear_numbers[i], *geom_coordinates[i]])

    self._mol = mol
    self._nuclear_numbers = nuclear_numbers
    self._geom_coordinates = geom_coordinates
    self._basis_set_name = basis_set_name
    self._ksdft_functional_name = ksdft_functional_name

    psi4.set_options({'basis': self._basis_set_name})
    wfn = psi4.core.Wavefunction.build(self._mol, psi4.core.get_global_option('BASIS'))
    self._mints = psi4.core.MintsHelper(wfn.basisset())


  def ao_kinetic_integral(self):
    return np.asarray(self._mints.ao_kinetic())


  def ao_hartree_potential_integral(self):
    return np.asarray(self._mints.ao_potential())


  def ap_electron_repulsion_integral(self):
    return np.asarray(self._mints.ao_eri())


  def ao_overlap_integral(self):
    return np.asarray(self._mints.ao_overlap())
