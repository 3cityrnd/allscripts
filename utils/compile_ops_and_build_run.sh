#!/bin/bash


P=`find result_dir |  grep -i  debug | head -n 1`


op_compiler -p $P -v Ascend910B3 -l debug -j 128 -o myout.run




