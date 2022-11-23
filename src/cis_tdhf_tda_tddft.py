import numpy as np

class driver():
  def __init__(self, scf_object):
    self._scf = scf_object

  def cis(self):
    # Not efficient but simple code

    if self._scf._spin_multiplicity != 1:
      raise NotImplementedError("Current CIS only support the singlet state.")

    # Set parameters
    num_occupied_mo = int(self._scf._num_electrons / 2)
    num_virtual_mo = self._scf.mo_coefficients.shape[1] - num_occupied_mo
    # mo_coeff (AO indexes, MO indexes)
    occupied_mo_coeff = self._scf.mo_coefficients[:, :num_occupied_mo]
    virtual_mo_coeff = self._scf.mo_coefficients[:, num_occupied_mo:]
    occupied_mo_energies = self._scf.mo_energies[:num_occupied_mo]
    virtual_mo_energies = self._scf.mo_energies[num_occupied_mo:]
    dim_cis_hamiltonian = num_occupied_mo * num_virtual_mo

    if dim_cis_hamiltonian > 1000:
      raise NotImplementedError(
          "The number of atomic orbitals is too much for the current CIS implementation.")

    ### Calculation of CIS Hamiltonian matrix in the MO basis
    # Indexes for MO integral
    # i, j: Occupied MOs
    # a, b: Virtural MOs
    # p, q, r, s: AOs
    # MO coefficients mo_coeff (AO indexes, MO indexes)

    orbital_energy_contributions = np.zeros(
        (dim_cis_hamiltonian, dim_cis_hamiltonian))
    # TODO: Demonstration of usefulness of Fortran module
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            if i == j and a == b:
              ia = i * num_virtual_mo + a
              jb = j * num_virtual_mo + b
              orbital_energy_contributions[ia, jb] = \
                  virtual_mo_energies[a] - occupied_mo_energies[i]

    # Calculate MO itnegrals in chemist's notation
    # AO electron repulsion integral is obtained in the SCF calculation
    # self._scf.ao_electron_repulsion_integral
    # Calculate (ia|jb)
    # This type of AO-MO integral conversion requires num_ao^8 scaling
    # and is not efficient.
    # The point-by-point conversions are required.
    mo_integral_iajb = np.einsum(
        'pi,qa,pqrs,rj,sb->iajb', occupied_mo_coeff, virtual_mo_coeff,
        self._scf.ao_electron_repulsion_integral, occupied_mo_coeff,
        virtual_mo_coeff, optimize=True)

    # Calculate (ij|ab)
    mo_integral_ijab = np.einsum(
        'pi,qj,pqrs,ra,sb->ijab', occupied_mo_coeff, occupied_mo_coeff,
        self._scf.ao_electron_repulsion_integral, virtual_mo_coeff,
        virtual_mo_coeff, optimize=True)

    # Prepare CIS Hamiltonian matrix
    cis_hamiltonian = np.zeros((dim_cis_hamiltonian, dim_cis_hamiltonian))

    # Calculate CIS Hamiltonian matrix
    # TODO: Demonstration of usefulness of Fortran module
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            ia = i * num_virtual_mo + a
            jb = j * num_virtual_mo + b
            cis_hamiltonian[ia, jb] = \
              orbital_energy_contributions[ia, jb] + \
                2.0 * mo_integral_iajb[i, a, j, b] - mo_integral_ijab[i, j, a, b]

    # Diagonalize the CIS Hamiltonian matrix
    # TODO: Use of the Davidson algorithm
    excitation_energies, eigen_vectors = np.linalg.eigh(cis_hamiltonian)

    print("")
    print("CIS excitation energies (eV):")
    print("The number of excited states is %s." % len(excitation_energies))
    au_to_ev = 27.21162
    print(*excitation_energies * au_to_ev)