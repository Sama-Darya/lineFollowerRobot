# Line Follower Robot
This repository provides a platform for online predictive learning for closed-loop systems, a line follower robot in this case. The physical robot implementation consists of two parts: The computer controller (on a Raspberry Pi) that generates the steering command and the arduino sketch that controls the robot's servo motors.

### Building RoboNet
RoboNet has the following dependencies that must be installed:
- ``boost``
- ``opencv``

In order to build:
- enter the LineFollowerRobot directory -- ``cd LineFollowerRobot``
- create a build directory -- ``mkdir build``
- run cmake -- ``cmake ..``
- run the build system -- ``make``

## Building ClBP
ClBP uses cmake. just enter the clBP directory from the root and type:
- ``mkdir build && cd build``
- ``cmake ..``
- ``make``
- record the path to both the generated library file (``libclBP.a``) and of the ``include`` directory.
