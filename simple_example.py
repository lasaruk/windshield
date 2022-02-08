import numpy
import windshield_camera

# Windshield parameters
n = numpy.array([0.0, -1.0, 1.0])
n = n / numpy.linalg.norm(n)
tau = 0.006
nu = 1.55

# Camera parameters
K = numpy.array([[1200.0, 0.0, 600.0], [0.0, 1200.0, 400.0], [0.0, 0.0, 1.0]])
r1 = -0.3
r2 = -0.2
R = numpy.diag([1.0, 1.0, 1.0])
t = numpy.array([0.0, 0.0, 0.0])

# Point to project
s = numpy.array([0.0, 1.0, 2.0])

# Central projection
p_central = windshield_camera.generalized_windshield_camera(s, K, windshield_camera.radial_distortions, [r1, r2],
                                                            windshield_camera.central_projection,
                                                            [], R, t)

# Exact projection
p_exact = windshield_camera.generalized_windshield_camera(s, K, windshield_camera.radial_distortions, [r1, r2],
                                                          windshield_camera.exact_windshield_projection,
                                                          [n, tau, nu], R, t)

# Approximate projection
p_approx = windshield_camera.generalized_windshield_camera(s, K, windshield_camera.radial_distortions, [r1, r2],
                                                           windshield_camera.approximate_windshield_projection,
                                                           [n, tau, nu], R, t)

print(p_central, p_exact, p_approx)
