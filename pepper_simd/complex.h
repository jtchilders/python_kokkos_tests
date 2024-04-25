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
   template<typename U,
          typename std::enable_if<std::is_same<MemSpace,U>::value,U>::type* = nullptr>
   ComplexView(const ComplexView<T,U>& lhs) :
      real(lhs.real),
      imag(lhs.imag)
   {}

   // copy constructor for non-like memory spaces
   template<typename U,
          typename std::enable_if<!std::is_same<MemSpace,U>::value,U>::type* = nullptr>
   ComplexView(const ComplexView<T,U>& lhs) :
      real(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.real)),
      imag(Kokkos::create_mirror_view_and_copy(MemSpace(), lhs.imag))
   {}


   // assign real/imag parts
   void assign(const int& i, const int& j, const T& re, const T& im) {
      real(i)[j] = re;
      imag(i)[j] = im;
   }

   void set_real(const int& i, const int& j, const T& re) {
      real(i)[j] = re;
   }
   void set_imag(const int& i, const int& j, const T& im) {
      imag(i)[j] = im;
   }
   void set(const int& i, const int& j, const T& re, const T& im) {
      real(i)[j] = re;
      imag(i)[j] = im;
   }

   // overload = operator
   ComplexView& operator=(const ComplexView& lhs) {
      real = lhs.real;
      imag = lhs.imag;
      return *this;
   }

};