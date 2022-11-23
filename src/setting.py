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
  mol_xyz = ''
  for line in xyz_lines:
    count += 1
    if count == 1:
      num_atoms = int(line.strip())
    elif count > 2:
      if len(line) == 0:
          break
      mol_xyz += str(line)
      parts = line.split()
      try:
          nuclear_numbers.append(int(parts[0]))
      except:
          nuclear_numbers.append(lut.element_Z_from_sym(parts[0]))
      coordinates.append([float(_) for _ in parts[1:4]])

  return mol_xyz, np.array(nuclear_numbers), np.array(coordinates)

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
  molecular_charge = sqc_conf['calc']['molecular_charge']
  spin_multiplicity = sqc_conf['calc']['spin_multiplicity']

  molecular_charge = int(molecular_charge)
  spin_multiplicity = int(spin_multiplicity)

  mol_xyz, nuclear_numbers, geom_coordinates = read_xyz(xyz_file_name)

  try:
    if (sqc_conf['calc']['excited_state']).lower() == 'cis':
      flag_cis = True
    else:
      flag_cis = False
  except:
    flag_cis = False

  try:
    if (sqc_conf['calc']['post_hartree-fock']).lower() == 'mp2':
      flag_mp2 = True
    else:
      flag_mp2 = False
  except:
    flag_mp2 = False

  return mol_xyz, nuclear_numbers, geom_coordinates, basis_set_name, \
    ksdft_functional_name, molecular_charge, spin_multiplicity, flag_cis, \
    flag_mp2