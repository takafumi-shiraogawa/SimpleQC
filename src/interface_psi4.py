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
    # psi4.set_options({'BASIS': self._basis_set_name,
    #               'DFT_SPHERICAL_POINTS': 6,
    #               'DFT_RADIAL_POINTS':    5})
    wfn = psi4.core.Wavefunction.build(self._mol, psi4.core.get_global_option('BASIS'))
    self._wfn = wfn
    self._psi4_object_ao_basis_sets = wfn.basisset()
    self._mints = psi4.core.MintsHelper(self._psi4_object_ao_basis_sets)

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


  def set_Vpot(self):
    self._Vpot = psi4.core.VBase.build(
        self._psi4_object_ao_basis_sets, self._psi4_object_ksdft_functional, "RV")
    self._Vpot.initialize()
    self._Vpot.initialize()


  def set_density_matrix_in_Vpot(self, density_matrix):
    self._Vpot.set_D([density_matrix])
    self._Vpot.properties()[0].set_pointers(density_matrix)


  # def ao_values(self):
  #   # at each grid
  #   svwn_w, wfn = psi4.energy("SVWN", return_wfn=True)
  #   Vpot = wfn.V_potential()

  #   # Grab a "points function" to compute the Phi matrices
  #   points_func = Vpot.properties()[0]

  #   # Grab a block and obtain its local mapping
  #   block = Vpot.get_block(1)
  #   npoints = block.npoints()
  #   lpos = np.array(block.functions_local_to_global())
  #   print("Local basis function mapping")
  #   print(lpos)
  #   print(len(lpos))

  #   # Copmute phi, note the number of points and function per phi changes.
  #   phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
  #   print("\nPhi Matrix")
  #   print(phi)
  #   print(np.shape(phi))
  #   phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
  #   print(np.shape(phi_x))

  #   # num_grids = len(grids)
  #   # for idx_grid in range(num_grids):
  #     # # compute_phi does not work!
  #     # ao_values[idx_grid] = self._psi4_object_ao_basis_sets.compute_phi(
  #     #     grids[idx_grid, :])

  #   # # Returns zero values
  #   # Vpot = psi4.core.VBase.build(
  #   #     self._psi4_object_ao_basis_sets, self._psi4_object_ksdft_functional, "RV")
  #   # Vpot.initialize()
  #   # # Grab a "points function" to compute the Phi matrices
  #   # points_func = Vpot.properties()[0]
  #   # # Grab a block and obtain its local mapping
  #   # block = Vpot.get_block(1)
  #   # npoints = block.npoints()
  #   # print(npoints)
  #   # lpos = np.array(block.functions_local_to_global())
  #   # print("Local basis function mapping")
  #   # print(lpos)
  #   # # Copmute phi, note the number of points and function per phi changes.
  #   # phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
  #   # print("\nPhi Matrix")
  #   # print(np.shape(phi))
  #   # print(phi)


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