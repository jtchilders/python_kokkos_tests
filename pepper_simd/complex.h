#include "Kokkos_Complex.hpp"
#include "Kokkos_SIMD.hpp"

/// This file will create a new Complex class that uses SIMD vectors and operators from the Kokkos library
// The class will contain a real and imaginary part
// each are Views of SIMD objects that will enable proper SIMD math operations are used

template<typename T>
using native_simd = Kokkos::Experimental::native_simd<T>;

template <typename T, typename MemSpace>
struct ComplexView {
   Kokkos::View<native_simd<T>*,Kokkos::DefaultExecutionSpace::array_layout,MemSpace> real;
   Kokkos::View<native_simd<T>*,Kokkos::DefaultExecutionSpace::array_layout,MemSpace> imag;

   // constructor
   ComplexView(const int n) :
      real(Kokkos::ViewAllocateWithoutInitializing("real"), n),
      imag(Kokkos::ViewAllocateWithoutInitializing("imag"), n)
   {}

   // copy constructor for like memory spaces
   template<typename T,
          typename std::enable_if<std::is_same<MemSpace,T>::value,T>::type* = nullptr>
   ComplexView(const ComplexView<T>& lhs) :
      real(lhs.real),
      imag(lhs.imag)
   {}

   // copy constructor for non-like memory spaces
   template<typename T,
          typename std::enable_if<!std::is_same<MemSpace,T>::value,T>::type* = nullptr>
   ComplexView(const ComplexView<T>& lhs) :
      real(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.real)),
      imag(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.imag))
   {}

};