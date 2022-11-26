import numpy as np

# kernel is not used in the current implementation
def lda_kernel(electron_density_at_grids):
  # exhange part
  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0))
  exchange_kernel = exchange_coeff * (electron_density_at_grids ** (1.0 / 3.0))

  # correlation part is neglected.

  return exchange_kernel


def lda_potential(electron_density_at_grids):
  # exhange part
  exchange_coeff = -(3.0 / np.pi) ** (1.0 / 3.0)
  exchange_potential = exchange_coeff * \
      (electron_density_at_grids ** (1.0 / 3.0))

  # correlation part is neglected.

  return exchange_potential


def lda_energy(electron_density_at_grids, grid_weights):
  # exhange part
  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0))
  exchange_energy = np.einsum(
      'n,n->', grid_weights, electron_density_at_grids ** (4.0 / 3.0))
  exchange_energy *= exchange_coeff

  # correlation part is neglected.

  return exchange_energy