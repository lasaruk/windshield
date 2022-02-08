import math
import numpy


def central_projection(s):
    # Implements the central projection of a space point to the z=1-plane.
    #
    # s Space point
    # Returns the central projection of s
    return numpy.array([s[0] / s[2], s[1] / s[2]])


def exact_windshield_projection(point, n, tau, nu):
    # Projects a point through the windshield by using the exact projection.
    #
    # point Space point
    # n Normal of the windshield
    # tau Thickness of the windshield
    # nu Refraction index of the windshield
    # Returns the result of central projection through the windshield
    w = numpy.dot(n, point)
    wsqr = w * w
    u = math.sqrt(numpy.dot(point, point) - w * w)
    usqr = u * u
    tsqr = tau * tau
    nusqr = nu * nu
    wwpuu = wsqr + usqr
    a4 = nusqr
    a3 = -2.0 * nusqr * (w + tau)
    a2 = (nusqr - 1.0) * (usqr + tsqr) + nusqr * w * (w + 4.0 * tau)
    a1 = -2.0 * tau * (nusqr * wwpuu + tau * w * (nusqr - 1.0) - usqr)
    a0 = (nusqr - 1.0) * tsqr * wwpuu
    delta0 = a2 * a2 - 3.0 * a3 * a1 + 12.0 * a4 * a0
    delta1 = 2.0 * a2 * a2 * a2 - 9.0 * a3 * a2 * a1 + 27.0 * a3 * a3 * a0 + 27.0 * a4 * a1 * a1 - 72.0 * a4 * a2 * a0
    omega = pow((delta1 + math.sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0)
    p = (8.0 * a4 * a2 - 3.0 * a3 * a3) / (8.0 * a4 * a4)
    s = math.sqrt(-2.0 * p / 3.0 + 1.0 / (3.0 * a4) * (omega + delta0 / omega)) / 2.0
    q = (a3 * a3 * a3 - 4.0 * a4 * a3 * a2 + 8.0 * a4 * a4 * a1) / (8.0 * a4 * a4 * a4)
    sigma = -a3 / (4.0 * a4) - s - math.sqrt(-4.0 * s * s - 2.0 * p + q / s) / 2.0
    return central_projection(point - sigma * n)


def approximate_windshield_projection(point, n, tau, nu):
    # Projects a point through the windshield by using the approximate projection.
    #
    # point Space point
    # n Normal of the windshield
    # tau Thickness of the windshield
    # nu Refraction index of the windshield
    # Returns the result of central projection through the windshield
    w = numpy.dot(n, point)
    wsqr = w * w
    u = math.sqrt(numpy.dot(point, point) - wsqr)
    usqr = u * u
    sigma = tau * (1.0 - 1.0 / math.sqrt((nu * nu - 1.0) * (usqr / wsqr + 1.0) + 1.0))
    return central_projection(point - sigma * n)


def radial_distortions(point, r1, r2):
    # Implements radial distortions.
    #
    # point Undistorted point in the z=1-plane
    # r1 First radial coefficient
    # r2 Second radial coefficient
    # Returns the distorted point
    r_sq = point[0] * point[0] + point[1] * point[1]
    return point * (1.0 + r1 * r_sq + r2 * r_sq * r_sq)


def generalized_windshield_camera(point, K, dist, dist_args, proj, proj_args, R, t):
    # Implements a generalized pinhole camera through the windshield.
    #
    # K First-order camera matrix containing the focal parameters, skew, and the principal point
    # dist Optional distortion function mapping an undistorted z=1-location u to a distorted one d(u)
    # dist_args Arguments of the distortion function
    # proj Projection function
    # proj_args Arguments of the projection function
    # R Rotation
    # t Translation
    # Returns the projected pixel coordinates
    return numpy.matmul(K, [*dist(proj(numpy.matmul(R, point - t), *proj_args), *dist_args), 1.0])
