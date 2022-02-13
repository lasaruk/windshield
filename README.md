# Windshield
Python code for a generalized pinhole camera model with windshield distortions

# Package summary
[camera_model.py](camera_model.py) contains an implementation of a generalized pinhole camera model suitable for optimization\
[slab_projection.py](slab_projection.py) extends the projection classes to an exact and approximate projection implementation\
[calibration.py](calibration.py) implements a calibration procedure of the model parameters given 3d-2d associations\
\
[simple_example.py](simple_example.py) demonstrates how to use the slab camera model for forward projection\
[backproject_example.py](backproject_example.py) demonstrates the accuracy of the methods\
[calibration_example.py](calibration_example.py) demonstrates how to calibrate the camera and the windshield