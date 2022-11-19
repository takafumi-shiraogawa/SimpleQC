from functools import wraps
import time
import setting as conf
import ksdft

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
def run_ksdft():
  nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name = conf.get_calc_params()
  calc_mol = ksdft.driver(nuclear_numbers, geom_coordinates,
                    basis_set_name, ksdft_functional_name)
  calc_mol.ks_scf()


if __name__ == "__main__":
  run_ksdft()