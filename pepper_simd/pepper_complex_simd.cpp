#include "Kokkos_Core.hpp"
#include "complex.h"
#include <iostream>

int main(int argc, char* argv[]) {

   // initialize kokkos
   Kokkos::ScopeGuard guard(argc, argv);
   Kokkos::print_configuration(std::cout, true);

   // parse first two arguments as nevents and nparticles or set default if nothing was passed
   int nevents = 100000;
   int nparticles = 1000;
   if(argc > 1)
      nevents = atoi(argv[1]);
   if(argc > 2)
      nparticles = atoi(argv[2]);
   std::cout << "nevents: " << nevents << std::endl;
   std::cout << "nparticles: " << nparticles << std::endl;

   ComplexView<double,Kokkos::HostSpace> cvA_host(nevents);
   ComplexView<double,Kokkos::HostSpace> cvB_host(nevents);

   for (int i = 0; i < nevents; i++) {
      for(int j = 0; j < native_simd<double>::size(); j++) {
         cvA_host.real(i)[j] = 2.0;
         cvA_host.imag(i)[j] = 2.0;
         cvB_host.real(i)[j] = 3.0;
         cvB_host.imag(i)[j] = -3.0;
      }
   }

   ComplexView<double,Kokkos::DefaultExecutionSpace> cvA_device(cvA_host);
   ComplexView<double,Kokkos::DefaultExecutionSpace> cvB_device(cvB_host);
   ComplexView<double,Kokkos::DefaultExecutionSpace> cvC_device(nevents);

   auto timer = Kokkos::Timer();
   // parallel for over events
   Kokkos::parallel_for(
      nevents,
      KOKKOS_LAMBDA(const int& i) {
         cvC_device.real(i) = cvA_device.real(i) + cvB_device.real(i);
         cvC_device.imag(i) = cvA_device.imag(i) + cvB_device.imag(i);

         ComplexView<double,Kokkos::DefaultExecutionSpace> cvD_device(5);

         for(int a = 0; a < 5; a++) {
            for(int b = 0; b < native_simd<double>::size(); b++) {
               cvD_device.set(a, b, a * b * 1.0, a * b * -1.0);
            }
         }
      }
   );
   // print time taken:
   std::cout << "Vector Time: " << timer.seconds() << std::endl;

   ComplexView<double,Kokkos::HostSpace> cvC_host(cvC_device);

   // reset timer
   timer.reset();

   // loop again but serially
   for (int i = 0; i < nevents; i++) {
      for(int j = 0; j < native_simd<double>::size(); j++) {
         cvC_host.real(i)[j] = cvA_host.real(i)[j] + cvB_host.real(i)[j];
         cvC_host.imag(i)[j] = cvA_host.imag(i)[j] + cvB_host.imag(i)[j];
      }
   }
   // print time taken:
   std::cout << "Serial time: " << timer.seconds() << std::endl;


   return 0;
}