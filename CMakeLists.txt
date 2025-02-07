cmake_minimum_required(VERSION 3.28)
project(mushroomclassificationann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCE_FILES
    src/main.cpp
    # src/Layer.cpp
    # src/ANN.cpp
    )

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0  # Change this to the latest stable version if needed
)

FetchContent_MakeAvailable(Eigen)

add_executable(main ${SOURCE_FILES})
target_link_libraries(main Eigen3::Eigen)
