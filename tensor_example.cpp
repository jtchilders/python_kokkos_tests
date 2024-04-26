#include <pybind11/pybind11.h>
#include <Kokkos_Core.hpp>
#include <iostream>

namespace py = pybind11;

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

struct TensorData {
    uintptr_t data_ptr;
    std::size_t size;
};

TensorData process_tensor(uintptr_t input_data_ptr, std::size_t size) {

   // Convert uintptr_t back to float* after passing from Python
   float* actual_data_ptr = reinterpret_cast<float*>(input_data_ptr);

   // Create an unmanaged Kokkos view from the raw pointer
   Kokkos::View<float*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_view(actual_data_ptr, size);

   // Allocate memory for output using Kokkos
   float* output_data_ptr = static_cast<float*>(Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>(size * sizeof(float)));

   // Create an unmanaged Kokkos view for the output
   Kokkos::View<float*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> output_view(output_data_ptr, size);

   // Perform operations on the GPU
   Kokkos::parallel_for("scale_data", Kokkos::RangePolicy<Kokkos::Cuda>(0, size), KOKKOS_LAMBDA(const int i) {
      output_view(i) = input_view(i) * 2.0f;  // Example operation
   });

   Kokkos::fence();  // Ensure Kokkos operations are completed

   // Return the output data pointer as uintptr_t
   TensorData result;
   result.data_ptr = reinterpret_cast<uintptr_t>(output_view.data());
   result.size = size;

   return result;
}

PYBIND11_MODULE(tenex, m) {

   // call the init and finalize functions
   m.def("init", &init);
   m.def("finalize", &finalize);
   m.def("print_hw_config", &print_hw_config);

   m.def("process_tensor", &process_tensor, "Process a tensor using Kokkos");
   
   // Register the TensorData structure
   py::class_<TensorData>(m, "TensorData")
      .def(py::init<>())  // Default constructor
      .def_readwrite("data_ptr", &TensorData::data_ptr, "Device pointer to the data")
      .def_readwrite("size", &TensorData::size, "Size of the data array");
    
}
