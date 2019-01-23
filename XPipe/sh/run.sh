#!/bin/sh

shopt -s expand_aliases
source ../../.bashrc



rank=$1
path=$2

thpython test_slurm.py -size 12 -path $path -rank $rank