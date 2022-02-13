# This example demonstrates the accuracy of the projection methods

import numpy

from slab_projection import *
from refractive_slab import *

# Slab parameters
elevation, azimuth = math.radians(10.0), 0.0
tau = 0.006
nu = 1.55
n = polar_normal(elevation, azimuth)

# Creating a slab model
refractive_slab = RefractiveSlab(PlanarInterface(n, -1.01, tau, nu), PlanarInterface(-n, 1.0, tau, nu))

# Projecting through the refractive slab into the scene
w = polar_normal(math.radians(5.0), 0.0)
s, u = refractive_slab.project(numpy.array([0.0, 0.0, 0.0]), w)
p, q = refractive_slab.project(s + u, -u)

# Projecting back
slab_projection = ApproximateSlabProjection(elevation, azimuth, tau, nu)
w_est = slab_projection(s + u)
print('Result', w / w[2], w_est)
