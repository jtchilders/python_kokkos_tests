# python_kokkos_tests
Testing how pybind11 can work with Kokkos C++ code.


# building

```bash
# clone repo
git clone --recursive git@github.com:jtchilders/python_kokkos_tests.git
```
## build Kokkos
```bash
cd python_kokkos_tests/extern/kokkos
# this fixed an issue on Nvidia GPUs.
git apply ../kokkos_4_3_00.patch
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$BASEPATH/python_kokkos_tests/install -DKokkos_ENABLE_OPENMP=1
make -C build -j install
cd ../..
export CMAKE_PREFIX_PATH=$PWD/install/lib/cmake/Kokkos
```


## build example
```bash
cmake -S . -B build
make -C build
export PYTHONPATH=$PWD/build
```

you'll need to have the `CMAKE_PREFIX_PATH` and `PYTHONPATH` set each time you login.
