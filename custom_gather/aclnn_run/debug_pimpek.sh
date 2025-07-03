#!/bin/bash

 

export ASCEND_GLOBAL_LOG_LEVEL=1

export ASCEND_PROCESS_LOG_PATH=`pwd`

export ASCEND_PROFILING_MODE=true

export PROFILING_OPTIONS="training_trace:on;task_trace:on;ai_core:on"

 

./launch_pimpe
