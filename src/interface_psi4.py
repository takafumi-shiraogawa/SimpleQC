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

    self._psi4_object_ao_basis_sets = wfn.basisset()
    self._psi4_object_ksdft_functional = psi4.driver.dft.build_superfunctional(
        'svwn', True)[0]
    # True means the restricted system


  def ao_kinetic_integral(self):
    return np.asarray(self._mints.ao_kinetic(), dtype='float64')


  def ao_hartree_potential_integral(self):
    return np.asarray(self._mints.ao_potential(), dtype='float64')


  def ap_electron_repulsion_integral(self):
    return np.asarray(self._mints.ao_eri(), dtype='float64')


  def ao_overlap_integral(self):
    return np.asarray(self._mints.ao_overlap(), dtype='float64')


  def gener_numerical_integral_grids_and_weights(self):
    # Generate 3d Cartesian grids and weights for each grid based on the Becke's method.
    # Refer to https://github.com/psi4/psi4numpy/blob/master/Tutorials/04_Density_Functional_Theory/4a_Grids.ipynb
    # written by Dr. Victor H. Chavez.
    Vpot = psi4.core.VBase.build(
        self._psi4_object_ao_basis_sets, self._psi4_object_ksdft_functional, "RV")
    Vpot.initialize()
    grids_x, grids_y, grids_z, weights = Vpot.get_np_xyzw()

    num_grids = len(weights)
    grids = np.zeros((num_grids, 3))
    grids[:, 0] = grids_x
    grids[:, 1] = grids_y
    grids[:, 2] = grids_z

    return grids, weights


  def check_basis_atomic_affiliation(self):
    # It is SO in Psi4
    num_ao = self._psi4_object_ao_basis_sets.nbf()
    ao_atomic_affiliation = np.zeros(num_ao, dtype=int)
    for i in range(num_ao):
      ao_atomic_affiliation[i] = self._psi4_object_ao_basis_sets.function_to_center(i)

    return ao_atomic_affiliation