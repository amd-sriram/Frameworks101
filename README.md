# Frameworks101

## Installation

> sh docker_run.sh

Inside the docker, run the following:

> cd /Frameworks
>
> sh install.sh
> 
> cd src


## First GPU Kernel

> cd FirstGPUKernel
>
> hipcc helloworld.cpp utils.hpp -o helloworld
> 
> ./helloworld
>
> hipcc add.cpp utils.hpp -o add
>
> ./add


## Pytorch

> python3 -m Pytorch.convolution
>
> python3 -m Pytorch.conv_back


## TorchCompile

To compare inference times of pytorch vs torch compile

> python3 TorchCompile/pytorch_model.py
>
> python3 TorchCompile/pytorch_compile.py

To generate the graphs

> python3 TorchCompile/compile_graphs.py

## ONNX

To export Resnet50 model

> python3 ONNX/export_model.py

To create ONNX models

> python3 ONNX/simple_onnx_model.py
>
> python3 ONNX/complex_onnx_model.py

To benchmark pytorch vs onnx for Resnet50 model

> python3 ONNX/benchmark.py