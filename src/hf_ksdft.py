import numpy as np
import psi4
import interface_psi4 as ipsi4


class driver():
  def __init__(self, nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name):
    self._nuclear_numbers = nuclear_numbers
    self._geom_coordinates = geom_coordinates
    self._basis_set_name = basis_set_name
    self._ksdft_functional_name = ksdft_functional_name
    self._num_electrons = nuclear_numbers.sum()


  @staticmethod
  def solve_one_electron_problem(orthogonalizer, fock_matrix):
    fock_matrix_in_orthogonalize_basis = np.matmul(
        np.transpose(orthogonalizer.conjugate()), fock_matrix)
    fock_matrix_in_orthogonalize_basis = np.matmul(
        fock_matrix_in_orthogonalize_basis, orthogonalizer)

    return np.linalg.eigh(fock_matrix_in_orthogonalize_basis)


  def calc_density_matrix_in_ao_basis(self, mo_coefficients):
    occupied_mo_coefficients = mo_coefficients[:, :int(self._num_electrons / 2)]

    return np.matmul(occupied_mo_coefficients, np.transpose(occupied_mo_coefficients))
    # It is equivalent to
    # density_matrix_in_ao_basis = np.einsum(
    #     'pi,qi->pq', occupied_mo_coefficients, occupied_mo_coefficients)
    # p, q: AO
    # i: MO

  @staticmethod
  def nuclei_nuclei(coordinates, charges):
    #: Conversion factor from Angstrom to Bohr
    ang_to_bohr = 1 / 0.52917721067
    natoms = len(coordinates)
    ret = 0.0
    for i in range(natoms):
        for j in range(i + 1, natoms):
            d = np.linalg.norm((coordinates[i] - coordinates[j]) *  ang_to_bohr)
            ret += charges[i] * charges[j] / d
    return ret


  def scf(self):
    # Not direct SCF

    # internal parameters
    num_max_scf_iter = 1000

    ### Preprocessing for SCF

    proc_ao_integral = ipsi4.proc_psi4(
        self._nuclear_numbers, self._geom_coordinates,
        self._basis_set_name, self._ksdft_functional_name)

    # Compute all requisite analytical AO integrations
    ao_kinetic_integral = proc_ao_integral.ao_kinetic_integral()
    ao_hartree_potential_integral = proc_ao_integral.ao_hartree_potential_integral()
    ao_electron_repulsion_integral = proc_ao_integral.ap_electron_repulsion_integral()
    ao_overlap_integral = proc_ao_integral.ao_overlap_integral()

    # Compute core_hamiltonian
    core_hamiltonian = ao_kinetic_integral + ao_hartree_potential_integral

    # Prepare the orthogonalizer S^(-1/2) for solving the one-electron eigenvalue problem
    overlap_eigen_value, overlap_eigen_function = np.linalg.eigh(
        ao_overlap_integral)
    # overlap_eigen_value is (num_ao)
    # overlap_eigen_function is (num_ao, num_ao)
    for i in range(len(overlap_eigen_value)):
      overlap_eigen_value[i] = overlap_eigen_value[i] ** (-0.5)
    half_inverse_overlap_eigen_value = np.diag(overlap_eigen_value)
    orthogonalizer = np.matmul(
        overlap_eigen_function, half_inverse_overlap_eigen_value)
    orthogonalizer = np.matmul(orthogonalizer, np.transpose(
        overlap_eigen_function.conjugate()))

    # Calculate initial guess of the density matrix by neglecting two-electron terms
    # of the Fock matrix
    # Solving the one-electron eigenvalue problem
    orbital_energies, mo_coefficients = driver.solve_one_electron_problem(
        orthogonalizer, core_hamiltonian)

    # Form MO coefficients in the original basis
    mo_coefficients = np.matmul(orthogonalizer, mo_coefficients)

    # Calculate density matrix in AO basis
    density_matrix_in_ao_basis = driver.calc_density_matrix_in_ao_basis(
        self, mo_coefficients)


    nuclear_repulsion_energy = driver.nuclei_nuclei(
        self._geom_coordinates, self._nuclear_numbers)


    ### Perform SCF
    for idx_scf in range(num_max_scf_iter):
      # Calculate Fock matrix in AO basis
      electron_repulsion_in_Fock_matrix = np.einsum(
          'pqrs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis)
      exchange_in_Fock_matrix = np.einsum(
          'prqs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis)
      fock_matrix = core_hamiltonian + 2.0 * electron_repulsion_in_Fock_matrix \
          - exchange_in_Fock_matrix

      # Calculate the electronic energy (without nuclei repulsion)
      electronic_energy = np.sum(np.multiply(
          density_matrix_in_ao_basis, core_hamiltonian + fock_matrix))
      total_energy = electronic_energy + nuclear_repulsion_energy
      print("SCF step %s: " % str(idx_scf + 1), total_energy, "Hartree")
      if idx_scf > 0:
        if abs(old_total_energy - total_energy) < 1.e-9:
          break
      old_total_energy = total_energy

      # Solving the one-electron eigenvalue problem
      orbital_energies, mo_coefficients = driver.solve_one_electron_problem(
        orthogonalizer, fock_matrix)

      # Form MO coefficients in the original basis
      mo_coefficients = np.matmul(orthogonalizer, mo_coefficients)

      # Calculate density matrix in AO basis
      density_matrix_in_ao_basis = driver.calc_density_matrix_in_ao_basis(
          self, mo_coefficients)