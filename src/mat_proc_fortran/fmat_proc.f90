subroutine fgener_cis_hamiltonian_matrix(num_occupied_mo, num_virtual_mo, &
  & orbital_energy_contributions, mo_integral_iajb, mo_integral_ijab, &
  & cis_hamiltonian)

  implicit none

  integer, intent(in) :: num_occupied_mo, num_virtual_mo
  double precision, intent(in) :: orbital_energy_contributions(:, :)
  double precision, intent(in) :: mo_integral_iajb(:, :, :, :)
  double precision, intent(in) :: mo_integral_ijab(:, :, :, :)

  real(8), intent(inout) :: cis_hamiltonian(:, :)

  integer :: i, j, a, b, idx_ia, idx_jb

  ! allocate(cis_hamiltonian(num_occupied_mo * num_virtual_mo, &
  !   & num_occupied_mo * num_virtual_mo))

  do i = 1, num_occupied_mo
    do j = 1, num_occupied_mo
      do a = 1, num_virtual_mo
        do b = 1, num_virtual_mo
          idx_ia = (i * (num_virtual_mo - 1)) + a
          idx_jb = (j * (num_virtual_mo - 1)) + b
          cis_hamiltonian(idx_ia, idx_jb) = &
              & orbital_energy_contributions(idx_ia, idx_jb) &
              & + 2.0 * mo_integral_iajb(i, a, j, b) &
              & - mo_integral_ijab(i, j, a, b)
        enddo
      enddo
    enddo
  enddo

end subroutine fgener_cis_hamiltonian_matrix