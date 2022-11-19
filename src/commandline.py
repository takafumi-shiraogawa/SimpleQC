import setting as conf
import ksdft

nuclear_numbers, geom_coordinates, basis_set_name, ksdft_functional_name = conf.get_calc_params()
calc_mol = ksdft.driver(nuclear_numbers, geom_coordinates,
                   basis_set_name, ksdft_functional_name)
calc_mol.ks_scf()