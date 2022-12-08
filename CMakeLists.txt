cmake_minimum_required(VERSION 3.1.0)
project(whisper)
set (CMAKE_CXX_STANDARD 20)

add_library(ggml SHARED "whisper.cpp/ggml.c")
find_package(pybind11 REQUIRED)


pybind11_add_module(whisperbind "../whisperbind.cpp")
target_link_libraries(whisperbind PUBLIC ggml)