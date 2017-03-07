#!/bin/bash

if [ "$1" = test ] ; then
    array_spec=0-1
    time_lim=1
    set -x
elif [ "$1" = prod ] ; then
    array_spec=0-1023
    time_lim=60
else
    echo >&2 "first argument must be either \"test\" or \"prod\""
    exit 1
fi

exec sbatch \
    -a $array_spec \
    -o '%A.out' \
    -J symphony \
    --mem=512 \
    -N 1 \
    -n 1 \
    -p serial_requeue \
    -t $time_lim \
    -o '%A.log' \
    --open-mode=append \
    $(dirname $0)/runner.sh $(cd $(dirname $0) && pwd)
