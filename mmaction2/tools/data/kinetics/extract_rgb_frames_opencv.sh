#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../
python build_rawframes.py ../../data/${DATASET}/videos_train_64/ ../../data/${DATASET}/rawframes_train/ --level 1  --ext mp4 --task rgb --new-short 64 --use-opencv
echo "Raw frames (RGB only) generated for train set"

#python build_rawframes.py ../../data/${DATASET}/videos_val_64/ ../../data/${DATASET}/rawframes_val/ --level 1 --ext mp4 --task rgb --new-short 64 --use-opencv
echo "Raw frames (RGB only) generated for val set"

cd ${DATASET}/
