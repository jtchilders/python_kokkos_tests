#include "Kokkos_Core.hpp"
#include "Kokkos_Complex.hpp"

// An example of building a class with Kokkos objects
// this class can be declared in different memory spaces
// then instances can be copied to other memory spaces.
// To ensure this copy is possible, we need to ensure
// member views are forced to use the layout of the
// DefaultExecutionSpace.
template <typename MemSpace> 
class Event_data {
   public:
   using kcx = Kokkos::complex<double>;
   using kv_fp64_2d_cx = Kokkos::View<kcx**, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;
   using kv_fp64_1d_cx = Kokkos::View<kcx*, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;
   using kv_fp64_2d = Kokkos::View<double**, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;
   using kv_fp64_1d = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;
   using kv_int64_2d = Kokkos::View<int**, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;
   using kv_int64_1d = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::array_layout, MemSpace>;

   int nevents;
   int nparticles;

   kv_fp64_2d particle_momenta;
   kv_fp64_1d event_weight;

   kv_int64_2d particle_parents;
   kv_int64_1d event_number;

   // generic constructor
   Event_data(const int nevents, const int nparticles) :
      nevents(nevents),
      nparticles(nparticles),
      particle_momenta(Kokkos::ViewAllocateWithoutInitializing("particle_momenta"), nevents, nparticles),
      event_weight(Kokkos::ViewAllocateWithoutInitializing("event_weight"), nevents),
      particle_parents(Kokkos::ViewAllocateWithoutInitializing("particle_parents"), nevents, nparticles),
      event_number(Kokkos::ViewAllocateWithoutInitializing("event_number"), nevents)
   {}

   // copy constructor for like-memory spaces
   template<typename T,
          typename std::enable_if<std::is_same<MemSpace,T>::value,T>::type* = nullptr>
   Event_data(const Event_data<T>& lhs):
      nevents(lhs.nevents),
      nparticles(lhs.nparticles),
      particle_momenta(lhs.particle_momenta),
      event_weight(lhs.event_weight),
      particle_parents(lhs.particle_parents),
      event_number(lhs.event_number)
   {}

   // copy constructor for non-like-memory spaces
   template<typename T,
          typename std::enable_if<!std::is_same<MemSpace,T>::value,T>::type* = nullptr>
   Event_data(const Event_data<T>& lhs):
      nevents(lhs.nevents),
      nparticles(lhs.nparticles),
      particle_momenta(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.particle_momenta)),
      event_weight(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.event_weight)),
      particle_parents(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.particle_parents)),
      event_number(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.event_number))
   {}

};