#!/bin/bash
SIGNAL_FILES=../raw_data/Xtohh*.root
BG_FILES=../raw_data/[!\(Xtohh\)\(data\)]*.root

for signal_file in $SIGNAL_FILES
do
    echo "Processing $signal_file file..."
    if [[ $signal_file =~ Xtohh([0-9]+).root ]]; then
        signal=${BASH_REMATCH[1]}
        hadd ../processed_data/all_${signal}.root $signal_file $BG_FILES
    else
        echo "Unable to parse string $signal_file"
    fi
done
