#!/bin/bash

#Arguments
#1 -> signal
#2 -> data location
#3 -> results location

echo $@

shopt -s extglob

signal_roots=$2/$1+([0-9]).root
all_roots=$2/*.root
bg_roots=""

for root in $all_roots
do
    if [[ ! $root =~ $1[0-9]+.root ]] && [[ ! $root =~ data.root ]]
    then
        bg_roots+="$root "
        echo "$root added to background list"
    fi
done

for root in $signal_roots
do
    echo -e "\nProcessing $root file..."
    if [[ $root =~ $1([0-9]+).root ]]
    then
        signal=${BASH_REMATCH[1]}
        hadd -f $3/all_${signal}.root $root $bg_roots
    else
        echo "Unable to parse string $root"
    fi
    
done