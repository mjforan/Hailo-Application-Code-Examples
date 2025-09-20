cmake_minimum_required(VERSION 3.16)
project(obj_det)

# Compilation options for warnings, debugging, and optimization
set(COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Werror -O3 -fconcepts)

# Find necessary packages
find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./*.cpp
    ./utils/*.cpp
)
list(APPEND SOURCES ../common/toolbox.cpp ../common/hailo_infer.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/utils ${CMAKE_CURRENT_SOURCE_DIR}/../common)

# Define the executable and link libraries
link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_compile_features(${PROJECT_NAME} PUBLIC c_std_99 cxx_std_20)  # Require C99 and C++20
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort rclcpp::rclcpp ${vision_msgs_TARGETS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
