# Defines projections through a parallel slab.

import math
import numpy

from camera_model import central_projection


def polar_normal(elevation, azimuth):
    # Defines the polar coordinate representation of a normal oriented towards the z-axis per default
    #
    # elevation Elevation of the normal vector
    # azimuth Azimuth of the normal vector
    # Returns a unit vector
    return numpy.array([-math.cos(elevation) * math.sin(azimuth), math.sin(elevation), math.cos(elevation) * math.cos(
        azimuth)])


class ExactSlabProjection:
    # Defines the exact central projection through a planar slab.

    def __init__(self, elevation=0.0, azimuth=0.0, tau=0.01, nu=1.0):
        # Initializes the projection
        #
        # elevation Elevation of the normal vector
        # azimuth Azimuth of the normal vector
        # tau Thickness of the slab
        # nu Refraction index
        self.elevation = elevation
        self.azimuth = azimuth
        self.tau = tau
        self.nu = nu

    def __call__(self, point):
        # Projects a point through the slab by using the exact projection.
        #
        # point Space point
        # Returns the result of central projection through the slab
        n = polar_normal(self.elevation, self.azimuth)
        w = numpy.dot(n, point)
        wsqr = w * w
        u = math.sqrt(numpy.dot(point, point) - w * w)
        usqr = u * u
        tsqr = self.tau * self.tau
        nusqr = self.nu * self.nu
        wwpuu = wsqr + usqr
        a4 = nusqr
        a3 = -2.0 * nusqr * (w + self.tau)
        a2 = (nusqr - 1.0) * (usqr + tsqr) + nusqr * w * (w + 4.0 * self.tau)
        a1 = -2.0 * self.tau * (nusqr * wwpuu + self.tau * w * (nusqr - 1.0) - usqr)
        a0 = (nusqr - 1.0) * tsqr * wwpuu
        delta0 = a2 * a2 - 3.0 * a3 * a1 + 12.0 * a4 * a0
        delta1 = 2.0 * a2 * a2 * a2 - 9.0 * a3 * a2 * a1 + 27.0 * a3 * a3 * a0 + 27.0 * a4 * a1 * a1 - 72.0 * a4 * a2 * a0
        omega = pow((delta1 + math.sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0)
        p = (8.0 * a4 * a2 - 3.0 * a3 * a3) / (8.0 * a4 * a4)
        s = math.sqrt(-2.0 * p / 3.0 + 1.0 / (3.0 * a4) * (omega + delta0 / omega)) / 2.0
        q = (a3 * a3 * a3 - 4.0 * a4 * a3 * a2 + 8.0 * a4 * a4 * a1) / (8.0 * a4 * a4 * a4)
        sigma = -a3 / (4.0 * a4) - s - math.sqrt(-4.0 * s * s - 2.0 * p + q / s) / 2.0
        return central_projection(point - sigma * n)

    def vectorize(self):
        # Returns a vector representation of the class
        return [self.elevation, self.azimuth, self.tau, self.nu]

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        self.elevation = x[0]
        self.azimuth = x[1]
        self.tau = x[2]
        self.nu = x[3]
        return x[4:]


class ApproximateSlabProjection:
    # Defines an approximate projection through a parallel slab.

    def __init__(self, elevation=0.0, azimuth=0.0, tau=0.01, nu=1.0):
        # Initializes the projection
        #
        # elevation Elevation of the normal vector
        # azimuth Azimuth of the normal vector
        # tau Thickness of the slab
        # nu Refraction index
        self.elevation = elevation
        self.azimuth = azimuth
        self.tau = tau
        self.nu = nu

    def __call__(self, point):
        # Projects a point through the slab by using the approximate projection.
        #
        # point Space point
        # Returns the result of central projection through the windshield
        n = polar_normal(self.elevation, self.azimuth)
        w = numpy.dot(n, point)
        wsqr = w * w
        u = math.sqrt(numpy.dot(point, point) - wsqr)
        usqr = u * u
        sigma = self.tau * (1.0 - 1.0 / math.sqrt((self.nu * self.nu - 1.0) * (usqr / wsqr + 1.0) + 1.0))
        return central_projection(point - sigma * n)

    def vectorize(self):
        # Returns a vector representation of the class
        return [self.elevation, self.azimuth, self.tau, self.nu]

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        self.elevation = x[0]
        self.azimuth = x[1]
        self.tau = x[2]
        self.nu = x[3]
        return x[4:]
