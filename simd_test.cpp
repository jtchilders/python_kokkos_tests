#include "Kokkos_Core.hpp"
#include "Kokkos_SIMD.hpp"
#include <iostream>

// create some tests of the simd functions

// define some types
using double_v = Kokkos::Experimental::native_simd<double>;
using int_v = Kokkos::Experimental::native_simd<std::int64_t>;

// define a function that takes a view of simd objects and fills it with random numbers
template<typename T>
void fill_random(Kokkos::View<T*, Kokkos::DefaultExecutionSpace> view) {
   // fill view with random numbers
   Kokkos::parallel_for("fill_random", view.size(), KOKKOS_LAMBDA(const int& i) {
      view(i) = rand() % 100;
   });
}

// define a function that takes two views of simd objects and does some various operations
template<typename T>
void run_tests(Kokkos::View<T*, Kokkos::DefaultExecutionSpace> view1, Kokkos::View<T*, Kokkos::DefaultExecutionSpace> view2) {

   // do some tests
   Kokkos::parallel_for("test1", view1.size(), KOKKOS_LAMBDA(const int& i) {
      
      // generate some code here that exposes the difference between simd and non-simd
      view1(i) = (view1(i) + view2(i)) / view1(i);
      view2(i) = (view1(i) + view2(i)) / view1(i);
      view1(i) = (view1(i) * view2(i)) / view1(i);
      view2(i) = (view1(i) * view2(i)) / view1(i);
      view1(i) *= (view1(i) + view2(i)) / view1(i);
      view2(i) *= (view1(i) + view2(i)) / view1(i);
      view1(i) /= (view1(i) * view2(i)) / view1(i);
      view2(i) /= (view1(i) * view2(i)) / view1(i);

      view1(i) = (view1(i) + view2(i)) / view1(i);
      view2(i) = (view1(i) + view2(i)) / view1(i);
      view1(i) = (view1(i) * view2(i)) / view1(i);
      view2(i) = (view1(i) * view2(i)) / view1(i);
      view1(i) *= (view1(i) + view2(i)) / view1(i);
      view2(i) *= (view1(i) + view2(i)) / view1(i);
      view1(i) /= (view1(i) * view2(i)) / view1(i);
      view2(i) /= (view1(i) * view2(i)) / view1(i);

      view1(i) = (view1(i) + view2(i)) / view1(i);
      view2(i) = (view1(i) + view2(i)) / view1(i);
      view1(i) = (view1(i) * view2(i)) / view1(i);
      view2(i) = (view1(i) * view2(i)) / view1(i);
      view1(i) *= (view1(i) + view2(i)) / view1(i);
      view2(i) *= (view1(i) + view2(i)) / view1(i);
      view1(i) /= (view1(i) * view2(i)) / view1(i);
      view2(i) /= (view1(i) * view2(i)) / view1(i);
      
      view1(i) = (view1(i) + view2(i)) / view1(i);
      view2(i) = (view1(i) + view2(i)) / view1(i);
      view1(i) = (view1(i) * view2(i)) / view1(i);
      view2(i) = (view1(i) * view2(i)) / view1(i);
      view1(i) *= (view1(i) + view2(i)) / view1(i);
      view2(i) *= (view1(i) + view2(i)) / view1(i);
      view1(i) /= (view1(i) * view2(i)) / view1(i);
      view2(i) /= (view1(i) * view2(i)) / view1(i);
   });

   
}



// main function
int main(int argc, char* argv[]) {

   // initialize kokkos
   Kokkos::ScopeGuard guard(argc, argv);
   
   // set view length using the command line
   int view_length = 1000;
   if(argc > 1)
      view_length = atoi(argv[1]);
   std::cout << "View length: " << view_length << std::endl;

   // create a view of simd objects and run some tests
   Kokkos::View<double_v*, Kokkos::DefaultExecutionSpace> view1("view1", view_length);
   Kokkos::View<double_v*, Kokkos::DefaultExecutionSpace> view2("view2", view_length);

   // fill the view with random numbers
   fill_random(view1);
   fill_random(view2);

   // track testing time
   Kokkos::Timer timer;

   // run some tests
   timer.reset();
   run_tests(view1, view2);
   // print testing time
   std::cout << "Testing time (double_v): " << timer.seconds() << " seconds" << std::endl;

   // copy device views to host and print top value:
   auto h_view1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view1);
   auto h_view2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view2);
   std::cout << "Top value: " << h_view1(0)[0] << " " << h_view2(0)[0] << std::endl;


   // create a view of double and run some tests
   Kokkos::View<double*, Kokkos::DefaultExecutionSpace> view3("view3", view_length);
   Kokkos::View<double*, Kokkos::DefaultExecutionSpace> view4("view4", view_length);

   // fill the view with random numbers
   fill_random(view3);
   fill_random(view4);

   timer.reset();
   run_tests(view3, view4);
   // print testing time
   std::cout << "Testing time (double): " << timer.seconds() << " seconds" << std::endl;

   // copy device views to host and print top value:
   auto h_view3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view3);
   auto h_view4 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view4);
   std::cout << "Top value: " << h_view3(0) << " " << h_view4(0) << std::endl;

}