cmake_minimum_required(VERSION 3.0.2)
project(object_detection_tpu)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  )


catkin_package()

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

