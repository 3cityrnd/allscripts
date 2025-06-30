#!/bin/bash

 

if [ -z "$ASCEND_TOOLKIT_HOME" ]; then

         echo "Error: source set_env.sh"

        exit

fi

 

 

NAME="pimpek_gather"

CONF="${NAME}.json"

ARCH="ai_core-ascend310P"

 

msopgen gen -i $CONF -f pytorch -c $ARCH -lan cpp -out $NAME



