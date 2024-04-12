#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// a class example which contains a static singleton of itself that should be used in multiple other functions
class SingletonExample {
   public:
   SingletonExample() = default;
   static SingletonExample& get() {
      static SingletonExample example;
      return example;
   }

   // some example variables and methods
   static int x;
   static int y;

   static void initialize() {
      get().x = 1;
      get().y = 2;
   }
   static void update() {
      get().x = 2;
      get().y = 4;
   }
   static void print() {
      std::cout << get().x << " " << get().y << std::endl;
   }
};

// define static members
int SingletonExample::x = 0;
int SingletonExample::y = 0;


// a function to print the variables in the singleton


// now the pybind module
PYBIND11_MODULE(classex, m) {

   // call the init and finalize functions
   m.def("init", &SingletonExample::initialize);
   m.def("update", &SingletonExample::update);
   m.def("print", &SingletonExample::print);

}