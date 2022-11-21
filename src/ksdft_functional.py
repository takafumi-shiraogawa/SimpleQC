import numpy as np

def lda_kernel(electron_density_at_grid):
  # exhange part
  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0))
  exchange_kernel = exchange_coeff * (electron_density_at_grid ** (1.0 / 3.0))

  # correlation part is neglected.

  return exchange_kernel