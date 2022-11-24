import numpy as np

class driver():
  def __init__(self, scf_object):
    self._scf = scf_object

  def mp2(self):
    # Not efficient but simple code

    if self._scf._spin_multiplicity != 1:
      raise NotImplementedError("Current MP2 only support the singlet state.")

    # Set parameters
    num_occupied_mo = int(self._scf._num_electrons / 2)
    num_virtual_mo = self._scf.mo_coefficients.shape[1] - num_occupied_mo
    # mo_coeff (AO indexes, MO indexes)
    occupied_mo_coeff = self._scf.mo_coefficients[:, :num_occupied_mo]
    virtual_mo_coeff = self._scf.mo_coefficients[:, num_occupied_mo:]
    occupied_mo_energies = self._scf.mo_energies[:num_occupied_mo]
    virtual_mo_energies = self._scf.mo_energies[num_occupied_mo:]
    dim_mp2 = num_occupied_mo * num_virtual_mo

    if dim_mp2 > 100:
      raise NotImplementedError(
          "The number of atomic orbitals is too much for the current MP2 implementation.")

    ### Calculation of MP2 energy in the MO basis
    # Indexes for MO integral
    # i, j: Occupied MOs
    # a, b: Virtural MOs
    # p, q, r, s: AOs
    # MO coefficients mo_coeff (AO indexes, MO indexes)

    orbital_energy_contributions = np.zeros(
        (num_occupied_mo, num_occupied_mo, num_virtual_mo, num_virtual_mo))
    # TODO: Demonstration of usefulness of Fortran module
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            orbital_energy_contributions[i, j, a, b] = \
              occupied_mo_energies[i] + occupied_mo_energies[j] \
                - virtual_mo_energies[a] - virtual_mo_energies[b]

    # Calculate MO itnegrals in chemist's notation
    # AO electron repulsion integral is obtained in the SCF calculation
    # self._scf.ao_electron_repulsion_integral
    # Calculate (ia|jb)
    # This is also required in CIS.
    # This type of AO-MO integral conversion requires num_ao^8 scaling
    # and is not efficient.
    # The point-by-point conversions are required.
    mo_integral_iajb = np.einsum(
        'pi,qa,pqrs,rj,sb->iajb', occupied_mo_coeff, virtual_mo_coeff,
        self._scf.ao_electron_repulsion_integral, occupied_mo_coeff,
        virtual_mo_coeff, optimize=True)

    # Calculate (ib|ja)
    mo_integral_ibja = np.einsum(
        'pi,qb,pqrs,rj,sa->ibja', occupied_mo_coeff, virtual_mo_coeff,
        self._scf.ao_electron_repulsion_integral, occupied_mo_coeff,
        virtual_mo_coeff, optimize=True)

    # Calculate MP2 correlation energy
    # TODO: Demonstration of usefulness of Fortran module
    mp2_correlation_energy = 0.0
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            mp2_correlation_energy += \
              (2.0 * mo_integral_iajb[i, a, j, b] * mo_integral_iajb[i, a, j, b] \
              - mo_integral_iajb[i, a, j, b] * mo_integral_ibja[i, b, j, a]) \
                / orbital_energy_contributions[i, j, a, b]

    # Calculate MP2 energy
    mp2_energy = self._scf.scf_energy + mp2_correlation_energy

    print("")
    print("MP2 energy (Hartree):", mp2_energy)
    print("MP2 correlation energy (Hartree):", mp2_correlation_energy)