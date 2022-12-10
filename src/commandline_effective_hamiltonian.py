from functools import wraps
import time
import setting as conf
import hf_ksdft
import cis_tdhf_tda_tddft
import mp2
import effective_hamiltonian

def stop_watch(func):
  """ Measure time """
  @wraps(func)
  def wrapper(*args, **kargs):
    start = time.time()

    result = func(*args, **kargs)

    elapsed_time = time.time() - start

    print("")
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return result
  return wrapper

@stop_watch
def run_hf_ksdft(flag_multiple_mols=False):
  mol_xyz, nuclear_numbers, geom_coordinates, basis_set_name, \
    ksdft_functional_name, molecular_charge, spin_multiplicity, \
    flag_cis, flag_mp2 = conf.get_calc_params(
        flag_multiple_mols=flag_multiple_mols)
  myscf = hf_ksdft.driver(mol_xyz, nuclear_numbers,
                    geom_coordinates, basis_set_name,
                    ksdft_functional_name, molecular_charge,
                    spin_multiplicity)
  myscf.scf()

  return myscf, flag_mp2, flag_cis, mol_xyz

@stop_watch
def run_cis(scf_object):
  mycis = cis_tdhf_tda_tddft.driver(scf_object)

  return mycis.cis()

@stop_watch
def run_mp2(scf_object):
  mymp2 = mp2.driver(scf_object)

  return mymp2.mp2()

if __name__ == "__main__":
  scf_1, flag_mp2_1, flag_cis_1, mol_xyz_1 = run_hf_ksdft()
  scf_2, flag_mp2_2, flag_cis_2, mol_xyz_2 = run_hf_ksdft(True)
  interaction_energy = effective_hamiltonian.coulomb_interaction(
      scf_1, scf_2, mol_xyz_1, mol_xyz_2)
  print("Interaction energy: %s Hartree" % str(interaction_energy))

  # if flag_mp2:
  #   mp2 = run_mp2(scf)
  # if flag_cis:
  #   cis = run_cis(scf)