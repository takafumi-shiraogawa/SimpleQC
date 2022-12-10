import interface_psi4 as ipsi4

def coulomb_interaction(scf_object_1, scf_object_2, mol_xyz_1, mol_xyz_2):
  return ipsi4.calc_ao_repulsion_two_mols(
      mol_xyz_1 + mol_xyz_2, scf_object_1, scf_object_2)