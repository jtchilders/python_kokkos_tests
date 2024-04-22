// main test file for testing the pepper event data object class using SIMD

#include "Kokkos_Core.hpp"
#include "evt_data.h"
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

   // create an event data object on the host and populate it
   Event_data<Kokkos::HostSpace> ed_host(nevents, nparticles);

   // populate the event data
   for(int i=0;i<nevents;i++) {
      ed_host.event_weight(i) = 1.0;
      ed_host.event_number(i) = i;
      for(int j=0;j<nparticles;j++) {
         ed_host.particle_momenta(i,j) = 1.0;
         ed_host.particle_parents(i,j) = j*1.0;
      }
   }

   // create an event data object on the device
   Event_data<Kokkos::DefaultExecutionSpace> ed_device(ed_host);

   // use copy constructor to create an event data object on the device
   ed_device = ed_host;

   // do some math on the device
   Kokkos::parallel_for(
      nevents,
      KOKKOS_LAMBDA(const int& i) {
         for(int j=0;j<nparticles;j++)
            ed_device.event_weight(i) += ed_device.particle_momenta(i,j);
      }
   );

}