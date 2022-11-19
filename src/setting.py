import os
import configparser
import numpy as np
from basis_set_exchange import lut

def read_xyz(xyz_file_name):
  with open(xyz_file_name) as fh:
    xyz_lines = fh.readlines()

  nuclear_numbers = []
  coordinates = []

  count = 0
  for line in xyz_lines:
    count += 1
    if count == 1:
      num_atoms = int(line.strip())
    elif count > 2:
      if len(line) == 0:
          break
      parts = line.split()
      try:
          nuclear_numbers.append(int(parts[0]))
      except:
          nuclear_numbers.append(lut.element_Z_from_sym(parts[0]))
      coordinates.append([float(_) for _ in parts[1:4]])

  return np.array(nuclear_numbers), np.array(coordinates)


def get_calc_params():
  """ Get parameters for the calculation.
  """
  is_file = os.path.isfile('sqc.conf')
  if not is_file:
    raise FileExistsError("sqc.conf does not exist.")

  sqc_conf = configparser.ConfigParser()
  sqc_conf.read('sqc.conf')

  xyz_file_name = sqc_conf['calc']['geom_xyz']
  basis_set_name = sqc_conf['calc']['gauss_basis_set']
  ksdft_functional_name = sqc_conf['calc']['ksdft_functional']

  nuclear_numbers, geom_coordinates = read_xyz(xyz_file_name)

  return nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name