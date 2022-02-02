#! /usr/bin/env bash

mkdir build && cd build
cmake -DARGO_BACKEND_MPI=ON -DARGO_BACKEND_SINGLENODE=ON -DARGO_TESTS=ON -DBUILD_DOCUMENTATION=ON -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc ../
make
