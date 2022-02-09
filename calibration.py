# This file contains the calibration function for cameras

import numpy
from scipy.optimize import least_squares


class Calibration:
    # Defines the calibration class.

    def __init__(self, space, image, model, fixed=None):
        # Initializes the calibration class
        #
        # space List of 3d points
        # image List of associated 2d image points
        # model Camera model
        self.space = space
        self.image = image
        self.model = model
        self.fixed = fixed

    def __call__(self):
        # Runs the optimization for the given number of iterations
        self.param = self.model.vectorize()
        result = least_squares(lambda x: self.objective(x), self.param, method='lm')
        self.model.unvectorize(result['x'])

    def objective(self, param):
        # Defines the objective function for the calibration
        fixed_param = param.copy()
        if self.fixed is not None:
            for i in self.fixed:
                fixed_param[i] = self.param[i]
        self.model.unvectorize(fixed_param)
        residual = numpy.array([])
        for space, image in zip(self.space, self.image):
            predict = self.model(space)
            residual = numpy.append(residual, image - predict)
        return residual
