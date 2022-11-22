from functools import wraps
import time
import setting as conf
import hf_ksdft

def stop_watch(func):
  """ Measure time """
  @wraps(func)
  def wrapper(*args, **kargs):
    start = time.time()

    result = func(*args, **kargs)

    elapsed_time = time.time() - start

    print("")
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    with open('elapsed_time.dat', 'w') as tfile:
      tfile.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return result
  return wrapper

@stop_watch
def run_hf_ksdft():
  mol_xyz, nuclear_numbers, geom_coordinates, basis_set_name, \
    ksdft_functional_name, molecular_charge, spin_multiplicity, \
      = conf.get_calc_params()
  calc_mol = hf_ksdft.driver(mol_xyz, nuclear_numbers,
                    geom_coordinates, basis_set_name,
                    ksdft_functional_name, molecular_charge,
                    spin_multiplicity)
  calc_mol.scf()


if __name__ == "__main__":
  run_hf_ksdft()
