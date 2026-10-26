cmake_minimum_required(VERSION 3.5)

project(a07)

# 默认的构建类型
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(ament_cmake REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach(dir config launch meshes urdf)
  install(DIRECTORY ${dir}/
    DESTINATION share/${PROJECT_NAME}/${dir})
endforeach(dir)

ament_package()
