#!/bin/bash

[[ $(whoami) != "root" ]] && echo "You must run this setup as root" && exit 1

function prompt_default(){
	local prompt="$1"
	local default="$2"
	local result=

	echo "$prompt (blank = default '$default')" 1>&2
	read result
	if [ -z $result ] ; then
		result="$default"
	fi
	echo $result
}


HOSTNAME=$(prompt_default "Enter container hostname" $(hostname)-adl)
DATA_DIR=$(prompt_default "Enter host data directory" /data)
SCRATCH_DIR=$(prompt_default "Enter host scratch directory" /scratch)
# SRC_DIR=$(prompt_default "Enter host src directory" $HOME/src)
PORTS="-p 8888:8888 -p 7002:7002 -p 7003:7003"

echo using SCRATCH_DIR: $SCRATCH_DIR
# echo using SRC_DIR $SRC_DIR
echo using DATA_DIR $DATA_DIR
echo using HOSTNAME $HOSTNAME

mkdir -p $DATA_DIR
mkdir -p $SCRATCH_DIR
# mkdir -p $SRC_DIR

echo "Creating alias:"
ALIAS="alias abyss-dl='nvidia-docker run --user docker -it --rm -v $SCRATCH_DIR:/scratch -v $DATA_DIR:/data -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix $PORTS --hostname $HOSTNAME abyss/dl'"
echo $ALIAS
# echo $ALIAS | tee -a /etc/skel/.bashrc | tee -a $HOME/.bashrc
echo "Run abyss-dl using alias 'abyss-dl (use new terminal or source ~/.bashrc)"
