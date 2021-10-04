cmake_minimum_required(VERSION 2.8.3)
project(ads_b_read)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  serial
  std_msgs
  sensor_msgs
  message_generation
)

add_definitions(-std=c++11)# -m64) #-Wall)

add_message_files(
  FILES
	DataStream_Request.msg
	Traffic_Report.msg
	Traffic_Report_Array.msg
	Status.msg
	Dynamic_Ownship.msg
	Static.msg
)

generate_messages(
  DEPENDENCIES
	std_msgs
)


catkin_package(
  CATKIN_DEPENDS
    serial
    std_msgs
    sensor_msgs
    message_runtime
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(ads_b_read_node src/ads_b_read_node.cpp)
add_executable(parse_node src/parse_node.cpp)
add_executable(log_node src/log_node.cpp)
add_executable(mavros_log_node src/mavros_log_node.cpp)

add_dependencies(ads_b_read_node ads_b_read_generate_messages_cpp)
add_dependencies(parse_node ads_b_read_generate_messages_cpp)
add_dependencies(log_node ads_b_read_generate_messages_cpp)

target_link_libraries(
  ads_b_read_node
  	${catkin_LIBRARIES}
)

target_link_libraries(
	parse_node
        ${catkin_LIBRARIES}
)

target_link_libraries(
	log_node
        ${catkin_LIBRARIES}
)

target_link_libraries(
	mavros_log_node
        ${catkin_LIBRARIES}
)

