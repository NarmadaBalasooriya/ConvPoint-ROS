#!/bin/bash

echo "catkin clean"
catkin clean -y

echo "catkin build"
catkin build

echo "source devel loc"
source ~/convpoint_ws/devel/setup.bash

echo "roslaunch rangenet_ros "
roslaunch convpoint_ros run_segmentation.launch