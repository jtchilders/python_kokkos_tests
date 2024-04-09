#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Kokkos_Core.hpp"

namespace py = pybind11;

// Alias for device and host Kokkos views of 2D arrays
using d_kvint1d = Kokkos::View<int*, Kokkos::DefaultExecutionSpace>;
using h_kvint1d = Kokkos::View<int*, Kokkos::HostSpace>;

// function to initialize kokkos
void init(){
   Kokkos::initialize();
}

// function to finalize kokkos
void finalize(){
   Kokkos::finalize();
}
template<typename T>
py::array_t<T> vectorMultiplyTemplate(py::array_t<T, py::array::c_style | py::array::forcecast> arr1, py::array_t<T, py::array::c_style | py::array::forcecast> arr2) {
   // Ensure arrays are of the same size
   if (arr1.size() != arr2.size()) {
      throw std::runtime_error("Arrays must be of the same size.");
   }

   auto buf1 = arr1.request();
   auto buf2 = arr2.request();
   const int size = arr1.size();

   // Example using a template Kokkos::View for T type
   // Note: Actual Kokkos usage would depend on T being a supported type
   Kokkos::View<T*, Kokkos::HostSpace> h_arr1(static_cast<T*>(buf1.ptr), size);
   Kokkos::View<T*, Kokkos::HostSpace> h_arr2(static_cast<T*>(buf2.ptr), size);
   
   auto d_arr1 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_arr1);
   auto d_arr2 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_arr2);
   Kokkos::View<T*, Kokkos::DefaultExecutionSpace> d_arr3("d_arr3", size);

   // Your computation here...
   // This example assumes you can perform similar operations on T type
   Kokkos::parallel_for("vectorMultiply", size, KOKKOS_LAMBDA(const int& i) {
      d_arr3(i) = d_arr1(i) * d_arr2(i);
   });

   // Create an unmanaged Kokkos view for the result
   py::array_t<T> result_np(size); // NumPy array for the result
   auto result_buf = result_np.request();
   Kokkos::View<T*, Kokkos::HostSpace> h_arr3_unmanaged(static_cast<T*>(result_buf.ptr), size);

   // Copy the result back to the unmanaged view
   Kokkos::deep_copy(h_arr3_unmanaged, d_arr3);

   return result_np;
}

// Wrapper functions for specific types
py::array_t<int> vectorMultiplyInt(py::array_t<int, py::array::c_style | py::array::forcecast> arr1, py::array_t<int, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyTemplate<int>(arr1, arr2);
}

py::array_t<float> vectorMultiplyFloat(py::array_t<float, py::array::c_style | py::array::forcecast> arr1, py::array_t<float, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyTemplate<float>(arr1, arr2);
}

py::array_t<double> vectorMultiplyDouble(py::array_t<double, py::array::c_style | py::array::forcecast> arr1, py::array_t<double, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyTemplate<double>(arr1, arr2);
}

// now a simply serial vector multiply
template<typename T>
py::array_t<T> vectorMultiplySerial(py::array_t<T, py::array::c_style | py::array::forcecast> arr1, py::array_t<T, py::array::c_style | py::array::forcecast> arr2) {
   if (arr1.size() != arr2.size()) {
      throw std::runtime_error("Arrays must be of the same size.");
   }
   auto* buf1 = reinterpret_cast<T*>(arr1.request().ptr);
   auto* buf2 = reinterpret_cast<T*>(arr2.request().ptr);
   const int size = arr1.size();

   // create output array
   py::array_t<T> result_np(size); // NumPy array for the result
   auto* result_buf = reinterpret_cast<T*>(result_np.request().ptr);

   // for loop over entries to fill output array
   for (int i = 0; i < size; ++i) {
      result_buf[i] = buf1[i] * buf2[i];
   }

   return result_np;
}

// Wrapper functions for specific types
py::array_t<int> vectorMultiplySerialInt(py::array_t<int, py::array::c_style | py::array::forcecast> arr1, py::array_t<int, py::array::c_style | py::array::forcecast> arr2) {
   return vectorMultiplySerial<int>(arr1, arr2);
}
py::array_t<float> vectorMultiplySerialFloat(py::array_t<float, py::array::c_style | py::array::forcecast> arr1, py::array_t<float, py::array::c_style | py::array::forcecast> arr2) {
   return vectorMultiplySerial<float>(arr1, arr2);
}
py::array_t<double> vectorMultiplySerialDouble(py::array_t<double, py::array::c_style | py::array::forcecast> arr1, py::array_t<double, py::array::c_style | py::array::forcecast> arr2) {
   return vectorMultiplySerial<double>(arr1, arr2);
}


PYBIND11_MODULE(vecmul, m) {
   
   m.doc() = "pybind11 example with Kokkos using vector multiplication"; // optional module docstring

   // call the init and finalize functions
   m.def("init", &init);
   m.def("finalize", &finalize);

   m.def("vector_multiply_int", &vectorMultiplyInt, "Int Vector multiplication using Kokkos");
   m.def("vector_multiply_float", &vectorMultiplyFloat, "Float Vector multiplication using Kokkos");
   m.def("vector_multiply_double", &vectorMultiplyDouble, "Double Vector multiplication using Kokkos");

   m.def("vector_multiply_serial_int", &vectorMultiplySerialInt, "Serial Vector multiplication using Kokkos");
   m.def("vector_multiply_serial_float", &vectorMultiplySerialFloat, "Serial Float Vector multiplication using Kokkos");
   m.def("vector_multiply_serial_double", &vectorMultiplySerialDouble, "Serial Double Vector multiplication using Kokkos");
}