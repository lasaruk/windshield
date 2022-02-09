import numpy

from camera_model import *
from slab_projection import *
from calibration import *

# Slab parameters
elevation, azimuth = 0.5, 0.0
tau = 0.006
nu = 1.55

# Camera parameters
K = numpy.array([[1200.0, 0.0, 600.0], [0.0, 1200.0, 400.0], [0.0, 0.0, 1.0]])
r1 = -0.3
r2 = -0.2
rotX, rotY, rotZ = 0.0, 0.0, 0.0
t = numpy.array([0.0, 0.0, 0.0])

# Camera model
reference_camera = CameraModel(Pixel(K), RadialDistortion(r1, r2),
                               ApproximateSlabProjection(elevation, azimuth, tau, nu),
                               EulerPose(rotX, rotY, rotZ, t))
calibrated_camera = CameraModel(Pixel(numpy.diag([1000.0, 1000.0, 1.0])), RadialDistortion(0.0, 0.0),
                                ApproximateSlabProjection(0.0, 0.0, 0.006, 1.55),
                                EulerPose(rotX, rotY, rotZ, t))

space = []
image = []
for x in numpy.arange(-1.0, 2.0, 1.0):
    for y in numpy.arange(-1.0, 2.0, 1.0):
        for z in numpy.arange(3.0, 5.0, 1.0):
            s = numpy.array([x, y, z])
            space.append(s)
            image.append(reference_camera(s))

calibration = Calibration(space, image, calibrated_camera, [9, 10])
calibration()

for i, (x, y) in enumerate(zip(reference_camera.vectorize(), calibrated_camera.vectorize())):
    print(i, '\t', "%0.3f" % x, '\t', "%0.3f" % y, '\t', "%0.10f" % (x - y))
