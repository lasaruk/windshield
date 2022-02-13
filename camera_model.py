# Defines the standard components of a camera model including
#   a pixel map,
#   a distortions,
#   a central projection, and
#   a pose functors.

import math
import numpy


def central_projection(s):
    # Implements the central projection of a space point to the z=1-plane.
    #
    # s Space point
    # Returns the central projection of s
    return numpy.array([s[0] / s[2], s[1] / s[2]])


def euler_angles(rot_x, rot_y, rot_z):
    # Creates a rotation matrix with the given Euler angles in X-Y-Z rotation order.
    #
    # rot_x Rotation around the X-axis
    # rot_y Rotation around the Y-axis
    # rot_z Rotation around the Z-axis
    # Returns the rotation matrix R_z(rot_z)*R_y(rot_y)*R_x(rot_x)
    rot_mat_x = numpy.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(rot_x), math.sin(rot_x)], [0.0, -math.sin(rot_x), math.cos(rot_x)]])
    rot_mat_y = numpy.array(
        [[math.cos(rot_y), 0.0, math.sin(rot_y)], [0.0, 1.0, 0.0], [-math.sin(rot_y), 0.0, math.cos(rot_y)]])
    rot_mat_z = numpy.array(
        [[math.cos(rot_z), math.sin(rot_z), 0.0], [-math.sin(rot_z), math.cos(rot_z), 0.0], [0.0, 0.0, 1.0]])
    return numpy.matmul(-rot_mat_z, numpy.matmul(-rot_mat_y, -rot_mat_x))


class Pixel:
    # Defines a pixel map

    def __init__(self, K=numpy.diag([1.0, 1.0, 1.0])):
        # Initializes the pixel map
        #
        # K Camera matrix of the form
        # [[fx, skew, px],
        #  [0,  fy,   py],
        #  [0,  0,     1]]
        self.K = K[0:2, 0:3]

    def __call__(self, p):
        # Maps z=1-plane to the image plane
        #
        # p Point in 2d on the z=1-plane
        # Returns the respective 2d point in the image
        return numpy.matmul(self.K, numpy.array([p[0], p[1], 1.0]))

    def vectorize(self):
        # Returns a vector representation of the class
        return numpy.array([self.K[0, 0], self.K[1, 1], self.K[0, 1], self.K[0, 2], self.K[1, 2]])

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        self.K = numpy.array([[x[0], x[2], x[3]], [0.0, x[1], x[4]]])
        return x[5:]


class CentralProjection:
    # Defines a central projection

    def __call__(self, s):
        # Implements the central projection of a space point to the z=1-plane.
        #
        # s Space point
        # Returns the central projection of s
        return central_projection(s)

    def vectorize(self):
        # Returns a vector representation of the class
        return []

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        return x


class RadialDistortion:
    # Defines the radial distortion map

    def __init__(self, r1=0.0, r2=0.0):
        # Initializes radial distortion
        #
        # r1 First radial coefficient
        # r2 Second radial coefficient
        self.r1 = r1
        self.r2 = r2

    def __call__(self, p):
        # Implements radial distortions.
        #
        # point Undistorted point in the z=1-plane
        # r1 First radial coefficient
        # r2 Second radial coefficient
        # Returns the distorted point
        r_sq = p[0] * p[0] + p[1] * p[1]
        return p * (1.0 + self.r1 * r_sq + self.r2 * r_sq * r_sq)

    def vectorize(self):
        # Returns a vector representation of the class
        return [self.r1, self.r2]

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        self.r1 = x[0]
        self.r2 = x[1]
        return x[2:]


class EulerPose:
    # Defines a pose based on Euler angles

    def __init__(self, rot_x=0.0, rot_y=0.0, rot_z=0.0, t=numpy.array([0.0, 0.0, 0.0])):
        # Initializes the pose
        #
        # rot_x Rotation around the X-axis
        # rot_y Rotation around the Y-axis
        # rot_z Rotation around the Z-axis
        # t Translation
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.t = t

    def __call__(self, s):
        # Transforms the point by the pose.
        #
        # s Input 3d point
        # Returns R(rotX, rotY, rotZ)(s-t)
        return numpy.matmul(euler_angles(self.rot_x, self.rot_y, self.rot_z), s - self.t)

    def vectorize(self):
        # Returns a vector representation of the class
        return [self.rot_x, self.rot_y, self.rot_z, self.t[0], self.t[1], self.t[2]]

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        self.rot_x = x[0]
        self.rot_y = x[1]
        self.rot_z = x[2]
        self.t = numpy.array([x[3], x[4], x[5]])
        return x[6:]


class CameraModel:
    # Defines a generalized pinhole camera model.

    def __init__(self, pixel, distortion, projection, pose):
        # Initializes a generalized camera
        #
        # pixel Pixel map, a functor from 2d to 2d
        # distortion Distortion map, a functor from 2d to 2d
        # projection Projection, a functor from 3d to 2d
        # pose Rotation and translation of the camera, a functor from 3d to 3d
        self.pixel = pixel
        self.distortion = distortion
        self.projection = projection
        self.pose = pose

    def __call__(self, space):
        # Maps a space point to a pixel by the current camera instance
        #
        # space Space 3d point in world coordinates
        # Returns the image 2d point
        return self.pixel(self.distortion(self.projection(self.pose(space))))

    def vectorize(self):
        # Returns the representation of the camera as a vector
        return numpy.append(self.pixel.vectorize(), numpy.append(self.distortion.vectorize(),
                                                                 numpy.append(self.projection.vectorize(),
                                                                              self.pose.vectorize())))

    def unvectorize(self, x):
        # Takes the first entries of the vector as own parametrization and returns the rest
        #
        # x Vector containing the parameterization of the class in its first entries
        # Returns the rest of the vector
        return self.pose.unvectorize(
            self.projection.unvectorize(self.distortion.unvectorize(self.pixel.unvectorize(x))))
