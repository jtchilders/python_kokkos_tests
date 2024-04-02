#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Kokkos_Core.hpp"

namespace py = pybind11;

// Alias for device and host Kokkos views of 2D arrays
using d_kvint2d = Kokkos::View<int**, Kokkos::DefaultExecutionSpace>;
using h_kvint2d = Kokkos::View<int**, Kokkos::HostSpace>;

// function to initialize kokkos
void init(){
   Kokkos::initialize();
}

// function to finalize kokkos
void finalize(){
   Kokkos::finalize();
}
// Function to perform matrix multiplication with adjustable inner dimensions
py::array_t<int> matrixMultiply(py::array_t<int, py::array::c_style | py::array::forcecast> a, py::array_t<int, py::array::c_style | py::array::forcecast> b) {
   // Ensure input arrays are 2D
   if (a.ndim() != 2 || b.ndim() != 2) {
      throw std::runtime_error("Input arrays must be 2D.");
   }
   
   // Ensure the inner dimensions match
   if (a.shape(1) != b.shape(0)) {
      throw std::runtime_error("Inner dimensions of A and B must match.");
   }

   auto bufA = a.request();
   auto bufB = b.request();
   const int m = a.shape(0);
   const int n = a.shape(1);
   const int p = b.shape(1);

   // Wrap the input NumPy arrays in host-side Kokkos views
   h_kvint2d h_A("h_A", m, n), h_B("h_B", n, p);
   std::copy(static_cast<int*>(bufA.ptr), static_cast<int*>(bufA.ptr) + bufA.size, h_A.data());
   std::copy(static_cast<int*>(bufB.ptr), static_cast<int*>(bufB.ptr) + bufB.size, h_B.data());

   // Copy host views to device views
   d_kvint2d d_A("d_A", m, n), d_B("d_B", n, p), d_C("d_C", m, p);
   Kokkos::deep_copy(d_A, h_A);
   Kokkos::deep_copy(d_B, h_B);

   // Perform matrix multiplication on the device
   Kokkos::parallel_for("matrix_multiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, p}), KOKKOS_LAMBDA(const int i, const int j) {
      int sum = 0;
      for (int k = 0; k < n; ++k) {
         sum += d_A(i, k) * d_B(k, j);
      }
      d_C(i, j) = sum;
   });

   // Copy result back to host
   h_kvint2d h_C("h_C", m, p);
   Kokkos::deep_copy(h_C, d_C);

   // Create a new NumPy array for the result
   py::array_t<int> result({m, p});
   auto bufC = result.request();
   std::copy(h_C.data(), h_C.data() + bufC.size, static_cast<int*>(bufC.ptr));

   return result;
}


PYBIND11_MODULE(matmul, m) {
   
   m.doc() = "pybind11 example with Kokkos using matrix multiplication"; // optional module docstring

   // call the init and finalize functions
   m.def("init", &init);
   m.def("finalize", &finalize);

   m.def("matrix_multiply", &matrixMultiply, "Matrix multiplication using Kokkos");
}
