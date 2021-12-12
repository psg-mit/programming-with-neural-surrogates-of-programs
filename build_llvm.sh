#!/usr/bin/env bash

git clone https://github.com/Ithemal/DiffTune
cd DiffTune/llvm-mca-parametric
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm
cmake --build . --target llvm-mca
