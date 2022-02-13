# This example shows a simple application of the slab model

import numpy

from camera_model import *
from slab_projection import *

# Slab parameters
elevation, azimuth = -math.radians(45), 0.0
tau = 0.006
nu = 1.55

# Camera parameters
K = numpy.array([[1200.0, 0.0, 600.0], [0.0, 1200.0, 400.0], [0.0, 0.0, 1.0]])
r1 = -0.3
r2 = -0.2
rotX, rotY, rotZ = 0.0, 0.0, 0.0
t = numpy.array([0.0, 0.0, 0.0])

# Camera model
regular_camera = CameraModel(Pixel(K), RadialDistortion(r1, r2), CentralProjection(), EulerPose(rotX, rotY, rotZ, t))
exact_slab_camera = CameraModel(Pixel(K), RadialDistortion(r1, r2), ExactSlabProjection(elevation, azimuth, tau, nu),
                                EulerPose(rotX, rotY, rotZ, t))
approx_slab_camera = CameraModel(Pixel(K), RadialDistortion(r1, r2),
                                 ApproximateSlabProjection(elevation, azimuth, tau, nu),
                                 EulerPose(rotX, rotY, rotZ, t))

# Point to project
s = numpy.array([0.0, 1.0, 2.0])
# Central projection
p_central = regular_camera(s)
# Exact projection
p_exact = exact_slab_camera(s)
# Approximate projection
p_approx = approx_slab_camera(s)

print(p_central, p_exact, p_approx)
