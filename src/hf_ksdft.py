import numpy as np
import interface_psi4 as ipsi4
import ksdft_functional as kdf


class driver():
  def __init__(self, mol_xyz, nuclear_numbers, geom_coordinates,
               basis_set_name, ksdft_functional_name,
               molecular_charge, spin_multiplicity):
    self._mol_xyz = mol_xyz
    self._nuclear_numbers = nuclear_numbers
    self._geom_coordinates = geom_coordinates
    self._basis_set_name = basis_set_name
    self._ksdft_functional_name = ksdft_functional_name
    self._num_electrons = np.sum(nuclear_numbers) - molecular_charge
    self._spin_multiplicity = spin_multiplicity

    if self._ksdft_functional_name == None:
      self._flag_ksdft = False
    else:
      if str(self._ksdft_functional_name).lower() == 'lda':
        self._flag_ksdft = True
      else:
        raise NotImplementedError(
        "Only the LDA exchange functional is implemented.")

    if self._flag_ksdft and self._spin_multiplicity != 1:
      raise NotImplementedError(
        "Unrestricted KS-DFT cannot be used."
      )

  @staticmethod
  def solve_one_electron_problem(orthogonalizer, fock_matrix):
    fock_matrix_in_orthogonalize_basis = np.matmul(
        np.transpose(orthogonalizer.conjugate()), fock_matrix)
    fock_matrix_in_orthogonalize_basis = np.matmul(
        fock_matrix_in_orthogonalize_basis, orthogonalizer)

    return np.linalg.eigh(fock_matrix_in_orthogonalize_basis)


  def calc_density_matrix_in_ao_basis(self, mo_coefficients):
    # Restricted calculation
    if self._spin_multiplicity == 1:
      occupied_mo_coefficients = mo_coefficients[:, :int(self._num_electrons / 2)]

      # return 2.0 * np.matmul(occupied_mo_coefficients, np.transpose(occupied_mo_coefficients))
      return 2.0 * np.einsum('pi,qi->pq', occupied_mo_coefficients, occupied_mo_coefficients)
      # It is equivalent to
      # density_matrix_in_ao_basis = np.einsum(
      #     'pi,qi->pq', occupied_mo_coefficients, occupied_mo_coefficients)
      # p, q: AO
      # i: MO

    # Unrestricted calculation
    else:
      # Calculate the number of alpha and beta electrons
      diff_num_alpha_and_beta_electrons = int(self._spin_multiplicity - 1)
      num_base_electrons = int((self._num_electrons - diff_num_alpha_and_beta_electrons) / 2)
      num_alpha_electrons = num_base_electrons + diff_num_alpha_and_beta_electrons
      num_beta_electrons = num_base_electrons

      alpha_occupied_mo_coefficients = mo_coefficients[0, :, :num_alpha_electrons]
      beta_occupied_mo_coefficients = mo_coefficients[1, :, :num_beta_electrons]

      num_ao = mo_coefficients.shape[1]
      density_matrix = np.zeros((2, num_ao, num_ao))
      density_matrix[0] = np.einsum('pi,qi->pq', alpha_occupied_mo_coefficients, \
        alpha_occupied_mo_coefficients)
      density_matrix[1] = np.einsum('pi,qi->pq', beta_occupied_mo_coefficients, \
        beta_occupied_mo_coefficients)

      return density_matrix

  @staticmethod
  def calc_nuclei_nuclei_repulsion_energy(coordinates, charges):
    #: Conversion factor from Angstrom to Bohr
    ang_to_bohr = 1 / 0.52917721067
    natoms = len(coordinates)
    ret = 0.0
    for i in range(natoms):
      for j in range(i + 1, natoms):
        d = np.linalg.norm((coordinates[i] - coordinates[j]) * ang_to_bohr)
        ret += charges[i] * charges[j] / d
    return ret


  def scf(self):
    # Not direct SCF

    # internal parameters
    num_max_scf_iter = 1000

    ### Preprocessing for SCF

    proc_ao_integral = ipsi4.proc_psi4(
      self._mol_xyz, self._nuclear_numbers,
      self._geom_coordinates, self._basis_set_name,
      self._ksdft_functional_name)

    # Compute all requisite analytical AO integrations
    ao_kinetic_integral = proc_ao_integral.ao_kinetic_integral()
    ao_hartree_potential_integral = proc_ao_integral.ao_hartree_potential_integral()
    ao_electron_repulsion_integral = proc_ao_integral.ap_electron_repulsion_integral()
    ao_overlap_integral = proc_ao_integral.ao_overlap_integral()

    if self._flag_ksdft:
      # In the practical implementation, this should be calculated on the fly.
      num_ao = len(ao_overlap_integral)
      if num_ao > 120:
        raise NotImplementedError(
          "The number of AOs is too large for the current implementation.")
      real_space_grids, weights_grids = \
        proc_ao_integral.gener_numerical_integral_grids_and_weights()
      num_grids = len(real_space_grids)
      ao_values_at_grids = np.zeros((num_grids, num_ao))
      ao_values_at_grids = proc_ao_integral.gener_ao_values_at_grids(real_space_grids)

    # Compute core_hamiltonian
    core_hamiltonian = ao_kinetic_integral + ao_hartree_potential_integral

    # Prepare the orthogonalizer S^(-1/2) for solving the one-electron eigenvalue problem
    overlap_eigen_value, overlap_eigen_function = np.linalg.eigh(
      ao_overlap_integral)

    # Compute the number of atomic orbitals
    num_ao = len(overlap_eigen_value)
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
    orbital_energies, temp_mo_coefficients = driver.solve_one_electron_problem(
      orthogonalizer, core_hamiltonian)

    # Form MO coefficients in the original basis
    if self._spin_multiplicity == 1:
      mo_coefficients = np.matmul(orthogonalizer, temp_mo_coefficients)
    else:
      mo_coefficients = np.zeros((2, num_ao, num_ao))
      mo_coefficients[0] = np.matmul(orthogonalizer, temp_mo_coefficients)
      mo_coefficients[1] = np.matmul(orthogonalizer, temp_mo_coefficients)

    # Calculate density matrix in AO basis
    density_matrix_in_ao_basis = driver.calc_density_matrix_in_ao_basis(
      self, mo_coefficients)


    nuclear_repulsion_energy = driver.calc_nuclei_nuclei_repulsion_energy(
      self._geom_coordinates, self._nuclear_numbers)

    if self._flag_ksdft:
      # ao_values_at_grids is in the dimension (num_grids, num_ao).
      electron_density_at_grids = np.zeros(num_grids)
      # n: index for the grids
      # p, q: indexes for AOs
      electron_density_at_grids = np.einsum(
        'pq,np,nq->n', density_matrix_in_ao_basis,
        ao_values_at_grids, ao_values_at_grids)

    ### Perform SCF
    for idx_scf in range(num_max_scf_iter):

      # print('The number of electrons from the grids:', np.einsum(
      #     'n,n->', weights_grids, electron_density_at_grids))

      if self._flag_ksdft:
        # Compute exchange-correlation potential
        exchange_correlation_potential = kdf.lda_potential(electron_density_at_grids)

        # # Compute exchange-correlation potential in the Fock matrix
        exchange_correlation_potential_in_Fock_matrix = np.zeros((num_ao, num_ao))
        exchange_correlation_potential_in_Fock_matrix = np.einsum(
          'n,np,n,nq->pq', weights_grids, ao_values_at_grids,
          exchange_correlation_potential, ao_values_at_grids)

      # Calculate Fock matrix in AO basis
      if self._spin_multiplicity == 1:
        electron_repulsion_in_Fock_matrix = np.einsum(
          'pqrs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis)
        if not self._flag_ksdft:
          exchange_in_Fock_matrix = np.einsum(
            'prqs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis)
          fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            - 0.5 * exchange_in_Fock_matrix
        else:
          fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            + exchange_correlation_potential_in_Fock_matrix
      else:
        electron_repulsion_in_Fock_matrix = np.einsum(
          'pqrs,rs->pq', ao_electron_repulsion_integral,
          density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1])
        # in alpha-spin orbital basis
        alpha_exchange_in_Fock_matrix = np.einsum(
          'prqs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis[0])
        alpha_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
          - alpha_exchange_in_Fock_matrix
        # in beta-spin orbital basis
        beta_exchange_in_Fock_matrix = np.einsum(
          'prqs,rs->pq', ao_electron_repulsion_integral, density_matrix_in_ao_basis[1])
        beta_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
          - beta_exchange_in_Fock_matrix

      # Calculate the electronic energy (without nuclei repulsion)
      # electronic_energy = np.matmul(density_matrix_in_ao_basis, core_hamiltonian + fock_matrix)
      if self._spin_multiplicity == 1:
        if not self._flag_ksdft:
          electronic_energy = 0.5 * \
            np.einsum('pq,pq->', density_matrix_in_ao_basis,
                      core_hamiltonian + fock_matrix)
          # np.sum(np.multiply(
          #     density_matrix_in_ao_basis, core_hamiltonian + fock_matrix))
        else:
          exchange_correlation_energy = kdf.lda_energy(
            electron_density_at_grids, weights_grids)
          electronic_energy = 0.5 * \
            np.einsum('pq,pq->', density_matrix_in_ao_basis,
                      2.0 * core_hamiltonian + electron_repulsion_in_Fock_matrix)
          electronic_energy += exchange_correlation_energy
      else:
        electronic_energy = 0.5 * \
          np.einsum('pq,pq->', density_matrix_in_ao_basis[0] + \
            density_matrix_in_ao_basis[1], core_hamiltonian)
        electronic_energy += 0.5 * \
          np.einsum('pq,pq->', density_matrix_in_ao_basis[0],
                    alpha_fock_matrix)
        electronic_energy += 0.5 * \
          np.einsum('pq,pq->', density_matrix_in_ao_basis[1],
                    beta_fock_matrix)

      total_energy = electronic_energy + nuclear_repulsion_energy
      print("SCF step %s: " % str(idx_scf + 1), total_energy, "Hartree")
      if idx_scf > 0:
        if abs(old_total_energy - total_energy) < 1.e-9:
          break
      old_total_energy = total_energy

      # Solving the one-electron eigenvalue problem
      if self._spin_multiplicity == 1:
        orbital_energies, mo_coefficients = driver.solve_one_electron_problem(
          orthogonalizer, fock_matrix)
      else:
        alpha_orbital_energies, alpha_mo_coefficients = driver.solve_one_electron_problem(
          orthogonalizer, alpha_fock_matrix)
        beta_orbital_energies, beta_mo_coefficients = driver.solve_one_electron_problem(
          orthogonalizer, beta_fock_matrix)

      # Form MO coefficients in the original basis
      if self._spin_multiplicity == 1:
        mo_coefficients = np.matmul(orthogonalizer, mo_coefficients)
      else:
        mo_coefficients = np.zeros((2, alpha_fock_matrix.shape[0],
                                    alpha_fock_matrix.shape[1]))
        mo_coefficients[0] = np.matmul(orthogonalizer, alpha_mo_coefficients)
        mo_coefficients[1] = np.matmul(orthogonalizer, beta_mo_coefficients)

      # Calculate density matrix in AO basis
      density_matrix_in_ao_basis = driver.calc_density_matrix_in_ao_basis(
        self, mo_coefficients)

      # Calculate the electron density at the grids
      if self._flag_ksdft:
        electron_density_at_grids = np.einsum(
          'pq,np,nq->n', density_matrix_in_ao_basis,
          ao_values_at_grids, ao_values_at_grids)

    ### Save the SCF results for the post-Hartree-Fock theories
    self.density_matrix_in_ao_basis = density_matrix_in_ao_basis
    self.num_ao = num_ao
    self.mo_energies = orbital_energies
    self.mo_coefficients = mo_coefficients
    self.scf_energy = total_energy
    self.ao_electron_repulsion_integral = ao_electron_repulsion_integral


    def calc_mulliken_atomic_charges(density_matrix_in_ao_basis, ao_overlap_integral):
      ao_atomic_affiliation = proc_ao_integral.check_basis_atomic_affiliation()
      if self._spin_multiplicity == 1:
        charge_matrix = np.matmul(density_matrix_in_ao_basis, ao_overlap_integral)
      else:
        charge_matrix = np.matmul(density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1],
                                  ao_overlap_integral)
        spin_diff_charge_matrix = np.matmul(density_matrix_in_ao_basis[0] - \
                                  density_matrix_in_ao_basis[1],
                                  ao_overlap_integral)

      num_atom = len(self._nuclear_numbers)
      mulliken_atomic_charges = np.zeros(num_atom)
      for i in range(len(ao_atomic_affiliation)):
        mulliken_atomic_charges[ao_atomic_affiliation[i]] += -charge_matrix[i, i]
      mulliken_atomic_charges += self._nuclear_numbers

      if self._spin_multiplicity != 1:
        spin_diff_mulliken_atomic_charges = np.zeros(num_atom)
        for i in range(len(ao_atomic_affiliation)):
          spin_diff_mulliken_atomic_charges[ao_atomic_affiliation[i]
                                            ] += spin_diff_charge_matrix[i, i]

      print("Mulliken atomic charges (|e|):", *mulliken_atomic_charges)
      if self._spin_multiplicity != 1:
        print("Mulliken atomic spin charges (|e|):", *spin_diff_mulliken_atomic_charges)

    calc_mulliken_atomic_charges(density_matrix_in_ao_basis, ao_overlap_integral)