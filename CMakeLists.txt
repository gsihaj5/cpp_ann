cmake_minimum_required(VERSION 3.28)
project(mushroomclassificationann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_GENERATOR Ninja)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCE_FILES
    src/main.cpp
    src/Layer.cpp
    src/ANN.cpp
    )

add_executable(main ${SOURCE_FILES})

