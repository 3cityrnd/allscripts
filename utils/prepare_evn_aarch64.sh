#!/bin/bash


conda create -n pyt311 python=3.11 && conda activate pyt311

pip3 install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py --user

pip3 install torch==2.1.0 pyyaml setuptools torch-npu==2.1.0.post12 torch-tb-profiler-ascend --user


#pip3 install torch==2.6.0 --user
#pip3 install torch_npu==2.6.0rc1 --user


conda create -n pyt37 python=3.7 &&  conda activate pyt37

pip3 install attrs cython numpy==1.21.6 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py --user



