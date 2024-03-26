#include <iostream>
#include <pybind11/pybind11.h>
#include "Kokkos_Core.hpp"

namespace py = pybind11;
using kvint = Kokkos::View<int*,Kokkos::DefaultExecutionSpace>;

// function to initialize kokkos
void init(){
   Kokkos::initialize();
}

// function to finalize kokkos
void finalize(){
   Kokkos::finalize();
}

PYBIND11_MODULE(example, m) {
   
   m.doc() = "pybind11 example with Kokkos"; // optional module docstring

   // call the init and finalize functions
   m.def("init", &init);
   m.def("finalize", &finalize);

   // create a pybind class for the kvint View object
   py::class_<kvint>(m, "KV_Int")
      // initialize the kvint object with an extent
      .def(py::init<std::string,const int>())
      // expose the extent method which takes an integer as an argument
      .def("extent", [](const kvint &v, int dim) { return v.extent(dim); })
      // provide get and set functions for the view
      .def("set", [](const kvint& v, int i, int val) { v(i) = val; })
      .def("get", [](const kvint& v, int i) { return v(i); });

}