import math
import numpy


class PlanarInterface:
    # Defines a planar refractive interface.
    #
    # The normal points into the direction of the incident medium.
    # The refractive fraction nu shall be defined as nu = nu_emergent/nu_incident.
    # If you want to flip the normal, do not forget to flip the sign of d.

    def __init__(self, n, d, tau, nu):
        # Initializes the interface
        #
        # n Normal
        # d Distance to the origin
        # tau Thickness of the slab
        # nu Refraction index
        self.n = n
        self.d = d
        self.tau = tau
        self.nu = nu

    def intersect(self, s, u):
        # Intersects a ray with the interface.
        #
        # s Starting point of the ray
        # u Direction of the ray
        # Returns the point of intersection
        la = -(self.d + numpy.matmul(self.n, s)) / numpy.matmul(self.n, u)
        return s + la * u, la

    def refract(self, u):
        # Refracts a ray on the interface.
        #
        # u Direction of the ray
        # Returns the refractive direction
        ntu = numpy.matmul(self.n, u)
        utu = numpy.matmul(u, u)
        if ntu >= 0.0:
            nu = 1.0 / self.nu
            v = nu * (u - ntu * self.n) + math.sqrt(
                (1.0 - nu * nu) * utu + nu * nu * ntu * ntu) * self.n
        else:
            nu = self.nu
            v = nu * (u - ntu * self.n) - math.sqrt(
                (1.0 - nu * nu) * utu + nu * nu * ntu * ntu) * self.n
        return v


class RefractiveSlab:
    # Defines a refractive slab

    def __init__(self, incident, emergent):
        # Initializes the refractive slab
        #
        # incident Incident interface
        # emergent Emergent interface
        self.incident = incident
        self.emergent = emergent

    def project(self, s, u):
        # Projects through the refractive slab.
        #
        # s Initial point
        # u Incident direction
        # Returns a pair of the emergent point and direction
        xi, li = self.incident.intersect(s, u)
        xe, le = self.emergent.intersect(s, u)
        if abs(li) < abs(le):
            # The ray hits the incident surface first
            v = self.incident.refract(u)
            xm, lm = self.emergent.intersect(xi, v)
            w = self.emergent.refract(v)
        else:
            # The ray hits the emergent surface first
            v = self.emergent.refract(u)
            xm, lm = self.incident.intersect(xe, v)
            w = self.incident.refract(v)
        return xm, w
