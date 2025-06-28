#!/bin/bash


#export ASCEND_GLOBAL_LOG_LEVEL=1
#export ASCEND_PROCESS_LOG_PATH=`pwd`
#export ASCEND_PROFILING_MODE=true
#export PROFILING_OPTIONS="training_trace:on;task_trace:on;ai_core:on" 

#python3 npu_check.py

echo "Start profiling $1"
msprof --application="python $1" --output="LOG4_$1" --ai-core=on --aicpu=off --llc-profiling=read  --analyze=on  --runtime-api=on   --sys-hardware-mem=on 


