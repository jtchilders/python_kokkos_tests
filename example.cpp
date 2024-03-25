#include <pybind11/pybind11.h>
#include "Kokkos_Core.hpp"

namespace py = pybind11;
using kvint = Kokkos::View<int*>;

int add(int i = 1, int j = 2) {
   return i + j;
}

void init(){
   Kokkos::initialize();
}

void finalize(){
   Kokkos::finalize();
}

PYBIND11_MODULE(example, m) {

   Kokkos::ScopeGuard ksg;

   kvint view("view", 10);
   
   m.doc() = "pybind11 example plugin"; // optional module docstring

   m.def("add", &add, "A function which adds two numbers",
   py::arg("i") = 1, py::arg("j") = 2);

   m.def("init", &init, "initialize the kokkos");
   m.def("finalize", &finalize, "finalize the kokkos");

   py::class_<kvint>(m, "ViewInt1D", py::buffer_protocol())
   .def_buffer([](kvint &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(int),                            /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
            1,                                      /* Number of dimensions */
            { m.extent(0) },                        /* Buffer dimensions */
            { sizeof(float) * m.extent(0)  }        /* Strides (in bytes) for each index */
        );
    });
}