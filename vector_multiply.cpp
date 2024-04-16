#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Kokkos_Core.hpp"
#include "Kokkos_SIMD.hpp"
#include <iostream>

namespace py = pybind11;

using double_v = Kokkos::Experimental::native_simd<double>;
using int_v = Kokkos::Experimental::native_simd<std::int64_t>;


// function to initialize kokkos
void init(){
   Kokkos::initialize();
}

void print_hw_config(){
   Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout, true);
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
   Kokkos::View<T*, Kokkos::DefaultExecutionSpace> d_arr3(Kokkos::ViewAllocateWithoutInitializing("d_arr3"), size);

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


template<typename T>
py::array_t<T> vectorMultiplyVVTemplate(py::array_t<T, py::array::c_style | py::array::forcecast> arr1, py::array_t<T, py::array::c_style | py::array::forcecast> arr2) {
   // Ensure arrays are of the same size
   if (arr1.size() != arr2.size()) {
      throw std::runtime_error("Arrays must be of the same size.");
   }

   auto buf1 = arr1.request();
   auto buf2 = arr2.request();
   const int size = arr1.size();

   // Example using a template Kokkos::View for T type
   // Note: Actual Kokkos usage would depend on T being a supported type
   Kokkos::View<Kokkos::Experimental::native_simd<T>*, Kokkos::HostSpace> h_arr1("h_arr1", size);
   Kokkos::View<Kokkos::Experimental::native_simd<T>*, Kokkos::HostSpace> h_arr2("h_arr2", size);
   auto* buf1_ptr = reinterpret_cast<T*>(buf1.ptr);
   auto* buf2_ptr = reinterpret_cast<T*>(buf2.ptr);
   for (int i = 0; i < size; ++i) {
      h_arr1(i) = buf1_ptr[i];
      h_arr2(i) = buf2_ptr[i];
      // std::cout << "buf1[" << i << "] = " << buf1_ptr[i] << ", buf2[" << i << "] = " << buf2_ptr[i] << " -> h_arr1[" << i << "] = " << h_arr1(i)[0] << ", h_arr2[" << i << "] = " << h_arr2(i)[0] <<  std::endl;
   }
   
   auto d_arr1 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_arr1);
   auto d_arr2 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_arr2);
   Kokkos::View<Kokkos::Experimental::native_simd<T>*, Kokkos::DefaultExecutionSpace> d_arr3(Kokkos::ViewAllocateWithoutInitializing("d_arr3"), size);

   // This example assumes you can perform similar operations on T type
   Kokkos::parallel_for("vectorMultiply", size, KOKKOS_LAMBDA(const int& i) {
      d_arr3(i) = d_arr1(i) * d_arr2(i);
   });

   // Create Kokkos View for result
   py::array_t<T> result_np(size); // NumPy array for the result
   auto result_buf = result_np.request();
   auto* result_host_buf = static_cast<T*>(result_buf.ptr);
   
   auto h_arr3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_arr3);

   // copy View to python array
   for (int i = 0; i < size; ++i) {
      result_host_buf[i] = h_arr3(i)[0];
   }

   return result_np;
}

// Wrapper functions for specific types
py::array_t<int> vectorMultiplyVVInt(py::array_t<int, py::array::c_style | py::array::forcecast> arr1, py::array_t<int, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyVVTemplate<int>(arr1, arr2);
}

py::array_t<float> vectorMultiplyVVFloat(py::array_t<float, py::array::c_style | py::array::forcecast> arr1, py::array_t<float, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyVVTemplate<float>(arr1, arr2);
}

py::array_t<double> vectorMultiplyVVDouble(py::array_t<double, py::array::c_style | py::array::forcecast> arr1, py::array_t<double, py::array::c_style | py::array::forcecast> arr2) {
    return vectorMultiplyVVTemplate<double>(arr1, arr2);
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
   m.def("print_hw_config", &print_hw_config);

   m.def("vector_multiply_int", &vectorMultiplyInt, "Int Vector multiplication using Kokkos");
   m.def("vector_multiply_float", &vectorMultiplyFloat, "Float Vector multiplication using Kokkos");
   m.def("vector_multiply_double", &vectorMultiplyDouble, "Double Vector multiplication using Kokkos");

   m.def("vector_multiply_serial_int", &vectorMultiplySerialInt, "Serial Vector multiplication using Kokkos");
   m.def("vector_multiply_serial_float", &vectorMultiplySerialFloat, "Serial Float Vector multiplication using Kokkos");
   m.def("vector_multiply_serial_double", &vectorMultiplySerialDouble, "Serial Double Vector multiplication using Kokkos");

   m.def("vector_multiply_vv_int", &vectorMultiplyVVInt, "Int Vector Vector multiplication using Kokkos");
   m.def("vector_multiply_vv_float", &vectorMultiplyVVFloat, "Float Vector Vector multiplication using Kokkos");
   m.def("vector_multiply_vv_double", &vectorMultiplyVVDouble, "Double Vector Vector multiplication using Kokkos");
}