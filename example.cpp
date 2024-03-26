#include <iostream>
#include <pybind11/pybind11.h>
#include "Kokkos_Core.hpp"

namespace py = pybind11;
using kvint = Kokkos::View<int*>;

int add(int i = 1, int j = 2) {
   return i + j;
}


PYBIND11_MODULE(example, m) {

   Kokkos::ScopeGuard ksg;

   kvint view("view", 10);
   
   m.doc() = "pybind11 example plugin"; // optional module docstring

   m.def("add", &add, "A function which adds two numbers",
   py::arg("i") = 1, py::arg("j") = 2);


}