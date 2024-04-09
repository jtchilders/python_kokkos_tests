cmake_minimum_required(VERSION 3.5...3.26)
project(pykos LANGUAGES CXX)


add_subdirectory(extern/pybind11)

find_package(Kokkos REQUIRED)

pybind11_add_module(pykos example.cpp)
target_link_libraries(pykos PRIVATE Kokkos::kokkos)

pybind11_add_module(matmul matrix_multiply.cpp)
target_link_libraries(matmul PRIVATE Kokkos::kokkos)

pybind11_add_module(vecmul vector_multiply.cpp)
target_link_libraries(vecmul PRIVATE Kokkos::kokkos)