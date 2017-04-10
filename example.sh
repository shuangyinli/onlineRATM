#!/bin/bash

#This is simple example how to use online ratm for training.

#The train set is split into 11 parts to simulate the large scale document stream.

#
#Check ../input/ to show the input files: training split data folder and a init.beta
#Check ./output to show the output.

make clean
echo
make
echo
rm -f ./output/*

echo

time ./onlineratm est onlinesetting.txt ./input/split/ ./output/ ./input/init.beta

echo