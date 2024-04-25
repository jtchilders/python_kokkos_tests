cmake_minimum_required(VERSION 3.5...3.26)
project(pykos LANGUAGES CXX)


add_subdirectory(extern/pybind11)

# added for JLSE V100 testing
find_package(CUDAToolkit)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
## might need to comment out if not using CUDA

find_package(Kokkos REQUIRED)


pybind11_add_module(pykos example.cpp)
target_link_libraries(pykos PRIVATE Kokkos::kokkos)

pybind11_add_module(matmul matrix_multiply.cpp)
target_link_libraries(matmul PRIVATE Kokkos::kokkos)

pybind11_add_module(vecmul vector_multiply.cpp)
target_link_libraries(vecmul PRIVATE Kokkos::kokkos)


pybind11_add_module(classex class_example.cpp)
target_link_libraries(classex PRIVATE Kokkos::kokkos)

add_executable(simd_test simd_test.cpp)
target_link_libraries(simd_test PRIVATE Kokkos::kokkos)

add_executable(pepper_simd pepper_simd/pepper_simd.cpp)
target_link_libraries(pepper_simd PRIVATE Kokkos::kokkos)

add_executable(pepper_complex pepper_simd/pepper_complex_simd.cpp)
target_link_libraries(pepper_complex PRIVATE Kokkos::kokkos)

include(GNUInstallDirs)

install(TARGETS pykos matmul vecmul classex
  COMPONENT python
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(TARGETS simd_test pepper_simd pepper_complex RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")