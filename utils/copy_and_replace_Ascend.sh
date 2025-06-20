#!/bin/bash

org="/usr/local/Ascend"

dst="Ascend"
CURR=`pwd`
u=`whoami`

function prepare() {
if [ ! -d $org ]; then
         echo "Error dirctory does not exist $org"
	exit 
fi

echo "Found $org"


if [ -d $dst ]; then
         echo "Error dirctory exists $dst, remove ?"
	exit 
fi

echo "Found $dst"



sudo cp -r $org $dst
sudo chown -R $u:$u $dst

}

prepare

S="$dst/ascend-toolkit/set_env.sh"

chmod u+w $S
my=`pwd`
to="$my/$dst"

sed -i s#${org}#${to}#g $S

echo "Done $to"



