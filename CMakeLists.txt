cmake_minimum_required(VERSION 3.5...3.26)
project(example LANGUAGES CXX)

add_subdirectory(extern/pybind11)
pybind11_add_module(example example.cpp)