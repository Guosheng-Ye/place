#!/bin/bash
EXPID=$1
EPOCH=$2
for epoch_num in `seq 9 ${EPOCH}`
do
    echo "${EXPID}_${epoch_num}"
done
